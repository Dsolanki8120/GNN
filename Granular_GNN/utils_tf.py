# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tensorflow ops and helpers useful to manipulate graphs.

This module contains utility functions to operate with `Tensor`s representations
of graphs, in particular:

  - `placeholders_from_data_dicts` and `placeholders_from_networkx`
     create placeholder structures to represent graphs;

  - `get_feed_dict` allow to create a `feed_dict` from a `graphs.GraphsTuple`
    containing numpy arrays and potentially, `None` values;

  - `data_dicts_to_graphs_tuple` converts between data dictionaries and
    `graphs.GraphsTuple`;

  - `fully_connect_graph_static` (resp. `fully_connect_graph_dynamic`) adds
    edges to a `graphs.GraphsTuple` in a fully-connected manner, in the case
    where the number of nodes per graph is known at graph construction time and
    is the same for all graphs (resp. only known at runtime and may depend on
    the graph);

  - `set_zero_node_features`, `set_zero_edge_features` and
    `set_zero_global_features` complete a `graphs.GraphsTuple` with a `Tensor`
    of zeros for the nodes, edges and globals;

  - `concat` batches `graphs.GraphsTuple` together (when using `axis=0`), or
    concatenates them along their data dimension;

  - `repeat` is a utility convenient to broadcast globals to edges or nodes of
    a graph;

  - `get_graph` indexes or slices a `graphs.GraphsTuple` to extract a subgraph
    or a subbatch of graphs;

  - `stop_gradients` stops the gradients flowing through a graph;

  - `identity` applies a `tf.identity` to every field of a graph;

  - `make_runnable_in_session` allows to run a graph containing `None` fields
    through a Tensorflow session.

The functions in these modules are able to deal with graphs containing `None`
fields (e.g. featureless nodes, featureless edges, or no edges).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from absl import logging
import graphs
import utils_np
import six
from six.moves import range
import tensorflow as tf  
import tree


NODES = graphs.NODES
EDGES = graphs.EDGES
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS


def _get_shape(tensor):
  """Returns the tensor's shape.

   Each shape element is either:
   - an `int`, when static shape values are available, or
   - a `tf.Tensor`, when the shape is dynamic.

  Args:
    tensor: A `tf.Tensor` to get the shape of.

  Returns:
    The `list` which contains the tensor's shape.
  """

  shape_list = tensor.shape.as_list()
  if all(s is not None for s in shape_list):
    return shape_list
  shape_tensor = tf.shape(tensor)
  return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]


def _build_placeholders_from_specs(dtypes,
                                   shapes,
                                   force_dynamic_num_graphs=True):
  """Creates a `graphs.GraphsTuple` of placeholders with `dtypes` and `shapes`.

  The dtypes and shapes arguments are instances of `graphs.GraphsTuple` that
  contain dtypes and shapes, or `None` values for the fields for which no
  placeholder should be created. The leading dimension the nodes and edges are
  dynamic because the numbers of nodes and edges can vary.
  If `force_dynamic_num_graphs` is True, then the number of graphs is assumed to
  be dynamic and all fields leading dimensions are set to `None`.
  If `force_dynamic_num_graphs` is False, then `N_NODE`, `N_EDGE` and `GLOBALS`
  leading dimensions are statically defined.

  Args:
    dtypes: A `graphs.GraphsTuple` that contains `tf.dtype`s or `None`s.
    shapes: A `graphs.GraphsTuple` that contains `list`s of integers,
      `tf.TensorShape`s, or `None`s.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.

  Returns:
    A `graphs.GraphsTuple` containing placeholders.

  Raises:
    ValueError: The `None` fields in `dtypes` and `shapes` do not match.
  """
  dct = {}
  for field in ALL_FIELDS:
    dtype = getattr(dtypes, field)
    shape = getattr(shapes, field)
    if dtype is None or shape is None:
      if not (shape is None and dtype is None):
        raise ValueError(
            "only one of dtype and shape are None for field {}".format(field))
      dct[field] = None
    elif not shape:
      raise ValueError("Shapes must have at least rank 1")
    else:
      shape = list(shape)
      if field not in [N_NODE, N_EDGE, GLOBALS] or force_dynamic_num_graphs:
        shape[0] = None

      dct[field] = tf.placeholder(dtype, shape=shape, name=field)

  return graphs.GraphsTuple(**dct)


def _placeholders_from_graphs_tuple(graph, force_dynamic_num_graphs=True):
  """Creates a `graphs.GraphsTuple` of placeholders that matches a numpy graph.

  Args:
    graph: A `graphs.GraphsTuple` that contains numpy data.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.

  Returns:
    A `graphs.GraphsTuple` containing placeholders.
  """
  graph_dtypes = graph.map(
      lambda v: tf.as_dtype(v.dtype) if v is not None else None, ALL_FIELDS)
  graph_shapes = graph.map(lambda v: list(v.shape) if v is not None else None,
                           ALL_FIELDS)
  return _build_placeholders_from_specs(
      graph_dtypes,
      graph_shapes,
      force_dynamic_num_graphs=force_dynamic_num_graphs)


def get_feed_dict(placeholders, graph):
  """Feeds a `graphs.GraphsTuple` of numpy arrays or `None` into `placeholders`.

  When feeding a fully defined graph (no `None` field) into a session, this
  method is not necessary as one can directly do:

  ```
  _ = sess.run(_, {placeholders: graph})
  ```

  However, if the placeholders contain `None`, the above construction would
  fail. This method allows to replace the above call by

  ```
  _ = sess.run(_, get_feed_dict(placeholders: graph))
  ```

  restoring the correct behavior.

  Args:
    placeholders: A `graphs.GraphsTuple` containing placeholders.
    graph: A `graphs.GraphsTuple` containing placeholder compatibale values,
      or `None`s.

  Returns:
    A dictionary with key placeholders and values the fed in values.

  Raises:
    ValueError: If the `None` fields in placeholders and `graph` do not exactly
      match.
  """
  feed_dict = {}
  for field in ALL_FIELDS:
    placeholder = getattr(placeholders, field)
    feed_value = getattr(graph, field)
    if placeholder is None or feed_value is None:
      if not (placeholder is None and feed_value is None):
        raise ValueError("Field {} should be `None` in either none or both of "
                         "the placeholders and feed values.".format(field))
    else:
      feed_dict[placeholder] = feed_value
  return feed_dict


def placeholders_from_data_dicts(data_dicts,
                                 force_dynamic_num_graphs=True,
                                 name="placeholders_from_data_dicts"):
  """Constructs placeholders compatible with a list of data dicts.

  Args:
    data_dicts: An iterable of data dicts containing numpy arrays.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.
    name: (string, optional) A name for the operation.

  Returns:
    An instance of `graphs.GraphTuple` placeholders compatible with the
      dimensions of the dictionaries in `data_dicts`.
  """
  with tf.name_scope(name):
    graph = data_dicts_to_graphs_tuple(data_dicts)
    return _placeholders_from_graphs_tuple(
        graph, force_dynamic_num_graphs=force_dynamic_num_graphs)


def placeholders_from_networkxs(graph_nxs,
                                node_shape_hint=None,
                                edge_shape_hint=None,
                                data_type_hint=tf.float32,
                                force_dynamic_num_graphs=True,
                                name="placeholders_from_networkxs"):
  """Constructs placeholders compatible with a list of networkx instances.

  Given a list of networkxs instances, constructs placeholders compatible with
  the shape of those graphs.

  The networkx graph should be set up such that, for fixed shapes `node_shape`,
   `edge_shape` and `global_shape`:
    - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
      tensor of shape `node_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
      tensor of shape `edge_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
      in which the edges will be sorted in the resulting `data_dict`;
    - `graph_nx.graph["features"] is a tensor of shape `global_shape` or `None`.

  Args:
    graph_nxs: A container of `networkx.MultiDiGraph`s.
    node_shape_hint: (iterable of `int` or `None`, default=`None`) If the graph
      does not contain nodes, the trailing shape for the created `NODES` field.
      If `None` (the default), this field is left `None`. This is not used if
      `graph_nx` contains at least one node.
    edge_shape_hint: (iterable of `int` or `None`, default=`None`) If the graph
      does not contain edges, the trailing shape for the created `EDGES` field.
      If `None` (the default), this field is left `None`. This is not used if
      `graph_nx` contains at least one edge.
    data_type_hint: (numpy dtype, default=`np.float32`) If the `NODES` or
      `EDGES` fields are autocompleted, their type.
    force_dynamic_num_graphs: A `bool` that forces the batch dimension to be
      dynamic. Defaults to `True`.
    name: (string, optional) A name for the operation.

  Returns:
    An instance of `graphs.GraphTuple` placeholders compatible with the
      dimensions of the graph_nxs.
  """
  with tf.name_scope(name):
    graph = utils_np.networkxs_to_graphs_tuple(graph_nxs, node_shape_hint,
                                               edge_shape_hint,
                                               data_type_hint.as_numpy_dtype)
    return _placeholders_from_graphs_tuple(
        graph, force_dynamic_num_graphs=force_dynamic_num_graphs)


def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked tensors (Tensorflow).

  When a set of tensors are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked tensor. This
  computes those offsets.

  Args:
    sizes: A 1D `Tensor` of the sizes per graph.
    repeats: A 1D `Tensor` of the number of repeats per graph.

  Returns:
    A 1D `Tensor` containing the index offset per graph.
  """
  sizes = tf.cast(tf.convert_to_tensor(sizes[:-1]), tf.int32)
  offset_values = tf.cumsum(tf.concat([[0], sizes], 0))
  return repeat(offset_values, repeats)


def _nested_concatenate(input_graphs, field_name, axis):
  """Concatenates a possibly nested feature field of a list of input graphs."""
  features_list = [getattr(gr, field_name) for gr in input_graphs
                   if getattr(gr, field_name) is not None]
  if not features_list:
    return None

  if len(features_list) < len(input_graphs):
    raise ValueError(
        "All graphs or no graphs must contain {} features.".format(field_name))

  name = "concat_" + field_name
  return tree.map_structure(lambda *x: tf.concat(x, axis, name), *features_list)


def concat(input_graphs, axis, name="graph_concat"):
  """Returns an op that concatenates graphs along a given axis.

  In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
  along `axis` (if a fields is `None`, the concatenation is just a `None`).
  If `axis` == 0, then the graphs are concatenated along the (underlying) batch
  dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
  are also concatenated together.
  If `axis` != 0, then there is an underlying assumption that the receivers,
  SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
  but this is not checked by this op.
  The graphs in `input_graphs` should have the same set of keys for which the
  corresponding fields is not `None`.

  Args:
    input_graphs: A list of `graphs.GraphsTuple` objects containing `Tensor`s
      and satisfying the constraints outlined above.
    axis: An axis to concatenate on.
    name: (string, optional) A name for the operation.

  Returns: An op that returns the concatenated graphs.

  Raises:
    ValueError: If `values` is an empty list, or if the fields which are `None`
      in `input_graphs` are not the same for all the graphs.
  """
  if not input_graphs:
    raise ValueError("List argument `input_graphs` is empty")
  utils_np._check_valid_sets_of_keys([gr._asdict() for gr in input_graphs])  # pylint: disable=protected-access
  if len(input_graphs) == 1:
    return input_graphs[0]

  with tf.name_scope(name):
    nodes = _nested_concatenate(input_graphs, NODES, axis)
    edges = _nested_concatenate(input_graphs, EDGES, axis)
    globals_ = _nested_concatenate(input_graphs, GLOBALS, axis)

    output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
    if axis != 0:
      return output
    n_node_per_tuple = tf.stack(
        [tf.reduce_sum(gr.n_node) for gr in input_graphs])
    n_edge_per_tuple = tf.stack(
        [tf.reduce_sum(gr.n_edge) for gr in input_graphs])
    offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
    n_node = tf.concat(
        [gr.n_node for gr in input_graphs], axis=0, name="concat_n_node")
    n_edge = tf.concat(
        [gr.n_edge for gr in input_graphs], axis=0, name="concat_n_edge")
    receivers = [
        gr.receivers for gr in input_graphs if gr.receivers is not None
    ]
    receivers = receivers or None
    if receivers:
      receivers = tf.concat(receivers, axis, name="concat_receivers") + offsets
    senders = [gr.senders for gr in input_graphs if gr.senders is not None]
    senders = senders or None
    if senders:
      senders = tf.concat(senders, axis, name="concat_senders") + offsets
    return output.replace(
        receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)


def stop_gradient(graph,
                  stop_edges=True,
                  stop_nodes=True,
                  stop_globals=True,
                  name="graph_stop_gradient"):
  """Stops the gradient flow through a graph.

  Args:
    graph: An instance of `graphs.GraphsTuple` containing `Tensor`s.
    stop_edges: (bool, default=True) indicates whether to stop gradients for
      the edges.
    stop_nodes: (bool, default=True) indicates whether to stop gradients for
      the nodes.
    stop_globals: (bool, default=True) indicates whether to stop gradients for
      the globals.
    name: (string, optional) A name for the operation.

  Returns:
    GraphsTuple after stopping the gradients according to the provided
    parameters.

  Raises:
    ValueError: If attempting to stop gradients through a field which has a
      `None` value in `graph`.
  """

  base_err_msg = "Cannot stop gradient through {0} if {0} are None"
  fields_to_stop = []
  if stop_globals:
    if graph.globals is None:
      raise ValueError(base_err_msg.format(GLOBALS))
    fields_to_stop.append(GLOBALS)
  if stop_nodes:
    if graph.nodes is None:
      raise ValueError(base_err_msg.format(NODES))
    fields_to_stop.append(NODES)
  if stop_edges:
    if graph.edges is None:
      raise ValueError(base_err_msg.format(EDGES))
    fields_to_stop.append(EDGES)

  with tf.name_scope(name):
    return graph.map(tf.stop_gradient, fields_to_stop)


def identity(graph, name="graph_identity"):
  """Pass each element of a graph through a `tf.identity`.

  This allows, for instance, to push a name scope on the graph by writing:
  ```
  with tf.name_scope("encoder"):
    graph = utils_tf.identity(graph)
  ```

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s. `None` values are passed
      through.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` `graphs_output` such that for any field `x` in NODES,
    EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, if `graph.x` was
    `None`, `graph_output.x` is `None`, and otherwise
    `graph_output.x = tf.identity(graph.x)`
  """
  non_none_fields = [k for k in ALL_FIELDS if getattr(graph, k) is not None]
  with tf.name_scope(name):
    return graph.map(tf.identity, non_none_fields)


def make_runnable_in_session(graph, name="make_graph_runnable_in_session"):
  """Allows a graph containing `None` fields to be run in a `tf.Session`.

  The `None` values of `graph` are replaced by `tf.no_op()`. This function is
  meant to be called just before a call to `sess.run` on a Tensorflow session
  `sess`, as `None` values currently cannot be run through a session.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s or `None` values.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` `graph_output` such that, for any field `x` in NODES,
    EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, and a Tensorflow session
    `sess`, if `graph.x` was `None`, `sess.run(graph_output)` is `None`, and
    otherwise
  """
  none_fields = [k for k in ALL_FIELDS if getattr(graph, k) is None]
  with tf.name_scope(name):
    return graph.map(lambda _: tf.no_op(), none_fields)


def repeat(tensor, repeats, axis=0, name="repeat", sum_repeats_hint=None):
  """Repeats a `tf.Tensor`'s elements along an axis by custom amounts.

  Equivalent to Numpy's `np.repeat`.
  `tensor and `repeats` must have the same numbers of elements along `axis`.

  Args:
    tensor: A `tf.Tensor` to repeat.
    repeats: A 1D sequence of the number of repeats per element.
    axis: An axis to repeat along. Defaults to 0.
    name: (string, optional) A name for the operation.
    sum_repeats_hint: Integer with the total sum of repeats in case it is
      known at graph definition time.

  Returns:
    The `tf.Tensor` with repeated values.
  """
  with tf.name_scope(name):
    if sum_repeats_hint is not None:
      sum_repeats = sum_repeats_hint
    else:
      sum_repeats = tf.reduce_sum(repeats)

    # This is TPU compatible.
    # Create a tensor consistent with output size indicating where the splits
    # between the different repeats are. For example:
    #   repeats = [2, 3, 6]
    # with cumsum(exclusive=True):
    #   scatter_indices = [0, 2, 5]
    # with scatter_nd:
    #   block_split_indicators = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    # cumsum(exclusive=False) - 1
    #   gather_indices =         [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    # Note that scatter_nd accumulates for duplicated indices, so for
    # repeats = [2, 0, 6]
    # scatter_indices = [0, 2, 2]
    # block_split_indicators = [1, 0, 2, 0, 0, 0, 0, 0]
    # gather_indices =         [0, 0, 2, 2, 2, 2, 2, 2]

    # Sometimes repeats may have zeros in the last groups. E.g.
    # for repeats = [2, 3, 0]
    # scatter_indices = [0, 2, 5]
    # However, the `gather_nd` only goes up to (sum_repeats - 1) index. (4 in
    # the example). And this would throw an error due to trying to index
    # outside the shape. Instead we let the scatter_nd have one more element
    # and we trim it from the output.
    scatter_indices = tf.cumsum(repeats, exclusive=True)
    block_split_indicators = tf.scatter_nd(
        indices=tf.expand_dims(scatter_indices, axis=1),
        updates=tf.ones_like(scatter_indices),
        shape=[sum_repeats + 1])[:-1]
    gather_indices = tf.cumsum(block_split_indicators, exclusive=False) - 1

    # An alternative implementation of the same, where block split indicators
    # does not have an indicator for the first group, and requires less ops
    # but requires creating a matrix of size [len(repeats), sum_repeats] is:
    # cumsum_repeats = tf.cumsum(repeats, exclusive=False)
    # block_split_indicators = tf.reduce_sum(
    #     tf.one_hot(cumsum_repeats, sum_repeats, dtype=tf.int32), axis=0)
    # gather_indices = tf.cumsum(block_split_indicators, exclusive=False)

    # Now simply gather the tensor along the correct axis.
    repeated_tensor = tf.gather(tensor, gather_indices, axis=axis)

    shape = tensor.shape.as_list()
    shape[axis] = sum_repeats_hint
    repeated_tensor.set_shape(shape)
    return repeated_tensor


def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-`None` NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-`None` RECEIVERS field;
  otherwise, it is set to 0.

  Args:
    data_dict: An input `dict`.

  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = tf.shape(dct[data_field])[0]
      else:
        dct[number_field] = tf.constant(0, dtype=tf.int32)
  return dct


def _to_compatible_data_dicts(data_dicts):
  """Convert the content of `data_dicts` to tensors of the right type.

  All fields are converted to `Tensor`s. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `tf.int32`.

  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and
      values either `None`s, or quantities that can be converted to `Tensor`s.

  Returns:
    A list of dictionaries containing `Tensor`s or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:
        dtype = tf.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        result[k] = tf.convert_to_tensor(v, dtype)
    results.append(result)
  return results


def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.

  Args:
    data_dicts: An iterable of data dictionaries with keys a subset of
      `GRAPH_DATA_FIELDS`, plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`.
      Every element of `data_dicts` has to contain the same set of keys.
      Moreover, the key `NODES` or `N_NODE` must be present in every element of
      `data_dicts`.

  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.

  Raises:
    ValueError: If two dictionaries in `data_dicts` have a different set of
      keys.
  """
  # Go from a list of dict to a dict of lists
  dct = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        dct[k].append(v)
      elif k not in dct:
        dct[k] = None
  dct = dict(dct)

  # Concatenate the graphs.
  for field, tensors in dct.items():
    if tensors is None:
      dct[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      dct[field] = tf.stack(tensors)
    else:
      dct[field] = tf.concat(tensors, axis=0)

  # Add offsets to the receiver and sender indices.
  if dct[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(dct[N_NODE], dct[N_EDGE])
    dct[RECEIVERS] += offset
    dct[SENDERS] += offset

  return dct


def _create_complete_edges_from_nodes_static(n_node, exclude_self_edges):
  """Creates complete edges for a graph with `n_node`.

  Args:
    n_node: (python integer) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

  Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
  """
  receivers = []
  senders = []
  n_edges = 0
  for node_1 in range(n_node):
    for node_2 in range(n_node):
      if not exclude_self_edges or node_1 != node_2:
        receivers.append(node_1)
        senders.append(node_2)
        n_edges += 1

  return {
      RECEIVERS: tf.constant(receivers, dtype=tf.int32),
      SENDERS: tf.constant(senders, dtype=tf.int32),
      N_EDGE: tf.constant([n_edges], dtype=tf.int32)
  }


def _create_complete_edges_from_nodes_dynamic(n_node, exclude_self_edges):
  """Creates complete edges for a graph with `n_node`.

  Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

  Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
  """
  rng = tf.range(n_node)
  receivers, senders = tf.meshgrid(rng, rng)
  n_edge = n_node * n_node

  if exclude_self_edges:
    ind = tf.cast(1 - tf.eye(n_node), bool)
    receivers = tf.boolean_mask(receivers, ind)
    senders = tf.boolean_mask(senders, ind)
    n_edge -= n_node

  receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
  senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
  n_edge = tf.reshape(n_edge, [1])

  return {RECEIVERS: receivers, SENDERS: senders, N_EDGE: n_edge}


def _validate_edge_fields_are_all_none(graph):
  if not all(getattr(graph, x) is None for x in [EDGES, RECEIVERS, SENDERS]):
    raise ValueError("Can only add fully connected a graph with `None`"
                     "edges, receivers and senders")


def fully_connect_graph_static(graph,
                               exclude_self_edges=False,
                               name="fully_connect_graph_static"):
  """Adds edges to a graph by fully-connecting the nodes.

  This method can be used if the number of nodes for each graph in `graph` is
  constant and known at graph building time: it will be inferred by dividing
  the number of nodes in the batch(the length of `graph.nodes`) by the number of
  graphs in the batch (the length of `graph.n_node`). It is an error to call
  this method with batches of graphs with dynamic or uneven sizes; in the latter
  case, the method may silently yield an incorrect result.

  Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

  Raises:
    ValueError: If any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
    ValueError: If the number of graphs (extracted from `graph.n_node` leading
      dimension) or number of nodes (extracted from `graph.nodes` leading
      dimension) is not known at construction time, or if the latter does not
      divide the former (observe that this is only a necessary condition for
      the constantness of the number of nodes per graph).
  """
  _validate_edge_fields_are_all_none(graph)

  num_graphs = graph.n_node.shape.as_list()[0]
  if num_graphs is None:
    raise ValueError("Number of graphs must be known at construction time when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  num_nodes = graph.nodes.shape.as_list()[0]
  if num_nodes is None:
    raise ValueError("Number of nodes must be known at construction time when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  if num_nodes % num_graphs != 0:
    raise ValueError("Number of nodes must be the same in all graphs when "
                     "using `fully_connect_graph_static`. Did you mean to use "
                     "`fully_connect_graph_dynamic`?")
  num_nodes_per_graph = num_nodes // num_graphs

  with tf.name_scope(name):
    one_graph_edges = _create_complete_edges_from_nodes_static(
        num_nodes_per_graph, exclude_self_edges)
    n_edges = num_nodes_per_graph * (num_nodes_per_graph - 1)
    if not exclude_self_edges:
      n_edges += num_nodes_per_graph

    all_graph_edges = {
        k: tf.tile(v, [num_graphs]) for k, v in six.iteritems(one_graph_edges)
    }
    offsets = [
        num_nodes_per_graph * i  # pylint: disable=g-complex-comprehension
        for i in range(num_graphs)
        for _ in range(n_edges)
    ]
    all_graph_edges[RECEIVERS] += offsets
    all_graph_edges[SENDERS] += offsets
    return graph.replace(**all_graph_edges)


def fully_connect_graph_dynamic(graph,
                                exclude_self_edges=False,
                                name="fully_connect_graph_dynamic"):
  """Adds edges to a graph by fully-connecting the nodes.

  This method does not require the number of nodes per graph to be constant,
  or to be known at graph building time.

  Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

  Raises:
    ValueError: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
  """
  _validate_edge_fields_are_all_none(graph)

  with tf.name_scope(name):

    def body(i, senders, receivers, n_edge):
      edges = _create_complete_edges_from_nodes_dynamic(graph.n_node[i],
                                                        exclude_self_edges)
      return (i + 1, senders.write(i, edges[SENDERS]),
              receivers.write(i, edges[RECEIVERS]),
              n_edge.write(i, edges[N_EDGE]))

    num_graphs = get_num_graphs(graph)
    loop_condition = lambda i, *_: tf.less(i, num_graphs)
    initial_loop_vars = [0] + [
        tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
        for _ in range(3)  # senders, receivers, n_edge
    ]
    _, senders_array, receivers_array, n_edge_array = tf.while_loop(
        loop_condition, body, initial_loop_vars, back_prop=False)

    n_edge = n_edge_array.concat()
    offsets = _compute_stacked_offsets(graph.n_node, n_edge)
    senders = senders_array.concat() + offsets
    receivers = receivers_array.concat() + offsets
    senders.set_shape(offsets.shape)
    receivers.set_shape(offsets.shape)

    receivers.set_shape([None])
    senders.set_shape([None])

    num_graphs = graph.n_node.get_shape().as_list()[0]
    n_edge.set_shape([num_graphs])

    return graph._replace(senders=senders, receivers=receivers, n_edge=n_edge)


def set_zero_node_features(graph,
                           node_size,
                           dtype=tf.float32,
                           name="set_zero_node_features"):
  """Completes the node state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    node_size: (int) the dimension for the created node features.
    dtype: (tensorflow type) the type for the created nodes features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the node field, which is a `Tensor` of shape
    `[number_of_nodes, node_size]`  where `number_of_nodes = sum(graph.n_node)`,
    with type `dtype`, filled with zeros.

  Raises:
    ValueError: If the `NODES` field is not None in `graph`.
    ValueError: If `node_size` is None.
  """
  if graph.nodes is not None:
    raise ValueError(
        "Cannot complete node state if the graph already has node features.")
  if node_size is None:
    raise ValueError("Cannot complete nodes with None node_size")
  with tf.name_scope(name):
    n_nodes = tf.reduce_sum(graph.n_node)
    return graph._replace(
        nodes=tf.zeros(shape=[n_nodes, node_size], dtype=dtype))


def set_zero_edge_features(graph,
                           edge_size,
                           dtype=tf.float32,
                           name="set_zero_edge_features"):
  """Completes the edge state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    edge_size: (int) the dimension for the created edge features.
    dtype: (tensorflow type) the type for the created edge features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the edge field, which is a `Tensor` of shape
    `[number_of_edges, edge_size]`, where `number_of_edges = sum(graph.n_edge)`,
    with type `dtype` and filled with zeros.

  Raises:
    ValueError: If the `EDGES` field is not None in `graph`.
    ValueError: If the `RECEIVERS` or `SENDERS` field are None in `graph`.
    ValueError: If `edge_size` is None.
  """
  if graph.edges is not None:
    raise ValueError(
        "Cannot complete edge state if the graph already has edge features.")
  if graph.receivers is None or graph.senders is None:
    raise ValueError(
        "Cannot complete edge state if the receivers or senders are None.")
  if edge_size is None:
    raise ValueError("Cannot complete edges with None edge_size")
  with tf.name_scope(name):
    senders_leading_size = graph.senders.shape.as_list()[0]
    if senders_leading_size is not None:
      n_edges = senders_leading_size
    else:
      n_edges = tf.reduce_sum(graph.n_edge)
    return graph._replace(
        edges=tf.zeros(shape=[n_edges, edge_size], dtype=dtype))


def set_zero_global_features(graph,
                             global_size,
                             dtype=tf.float32,
                             name="set_zero_global_features"):
  """Completes the global state of a graph.

  Args:
    graph: A `graphs.GraphsTuple` with a `None` global state.
    global_size: (int) the dimension for the created global features.
    dtype: (tensorflow type) the type for the created global features.
    name: (string, optional) A name for the operation.

  Returns:
    The same graph but for the global field, which is a `Tensor` of shape
    `[num_graphs, global_size]`, type `dtype` and filled with zeros.

  Raises:
    ValueError: If the `GLOBALS` field of `graph` is not `None`.
    ValueError: If `global_size` is not `None`.
  """
  if graph.globals is not None:
    raise ValueError(
        "Cannot complete global state if graph already has global features.")
  if global_size is None:
    raise ValueError("Cannot complete globals with None global_size")
  with tf.name_scope(name):
    n_graphs = get_num_graphs(graph)
    return graph._replace(
        globals=tf.zeros(shape=[n_graphs, global_size], dtype=dtype))


def data_dicts_to_graphs_tuple(data_dicts, name="data_dicts_to_graphs_tuple"):
  """Creates a `graphs.GraphsTuple` containing tensors from data dicts.

   All dictionaries must have exactly the same set of keys with non-`None`
   values associated to them. Moreover, this set of this key must define a valid
   graph (i.e. if the `EDGES` are `None`, the `SENDERS` and `RECEIVERS` must be
   `None`, and `SENDERS` and `RECEIVERS` can only be `None` both at the same
   time). The values associated with a key must be convertible to `Tensor`s,
   for instance python lists, numpy arrays, or Tensorflow `Tensor`s.

   This method may perform a memory copy.

   The `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to
   `np.int32` type.

  Args:
    data_dicts: An iterable of data dictionaries with keys in `ALL_FIELDS`.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphTuple` representing the graphs in `data_dicts`.
  """
  data_dicts = [dict(d) for d in data_dicts]
  for key in ALL_FIELDS:
    for data_dict in data_dicts:
      data_dict.setdefault(key, None)
  utils_np._check_valid_sets_of_keys(data_dicts)  # pylint: disable=protected-access
  with tf.name_scope(name):
    data_dicts = _to_compatible_data_dicts(data_dicts)
    return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))


def _check_valid_index(index, element_name):
  """Verifies if a value with `element_name` is a valid index."""
  if isinstance(index, int):
    return True
  elif isinstance(index, tf.Tensor):
    if index.dtype != tf.int32 and index.dtype != tf.int64:
      raise TypeError(
          "Invalid tensor `{}` parameter. Valid tensor indices must have "
          "types tf.int32 or tf.int64, got {}."
          .format(element_name, index.dtype))
    if index.shape.as_list():
      raise TypeError(
          "Invalid tensor `{}` parameter. Valid tensor indices must be scalars "
          "with shape [], got{}"
          .format(element_name, index.shape.as_list()))
    return True
  else:
    raise TypeError(
        "Invalid `{}` parameter. Valid tensor indices must be integers "
        "or tensors, got {}."
        .format(element_name, type(index)))


def get_graph(input_graphs, index, name="get_graph"):
  """Indexes into a graph.

  Given a `graphs.graphsTuple` containing `Tensor`s and an index (either
  an `int` or a `slice`), index into the nodes, edges and globals to extract the
  graphs specified by the slice, and returns them into an another instance of a
  `graphs.graphsTuple` containing `Tensor`s.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing `Tensor`s.
    index: An `int`, a `slice`, a tensor `int` or a tensor `slice`, to index
      into `graph`. `index` should be compatible with the number of graphs in
      `graphs`. The `step` parameter of the `slice` objects must be None.
    name: (string, optional) A name for the operation.

  Returns:
    A `graphs.GraphsTuple` containing `Tensor`s, made of the extracted
      graph(s).

  Raises:
    TypeError: if `index` is not an `int`, a `slice`, or corresponding tensor
      types.
    ValueError: if `index` is a slice and `index.step` if not None.
  """

  def safe_slice_none(value, slice_):
    if value is None:
      return value
    return value[slice_]

  if isinstance(index, (int, tf.Tensor)):
    _check_valid_index(index, "index")
    graph_slice = slice(index, index + 1)
  elif (isinstance(index, slice) and
        _check_valid_index(index.stop, "index.stop") and
        (index.start is None or _check_valid_index(
            index.start, "index.start"))):
    if index.step is not None:
      raise ValueError("slices with step/stride are not supported, got {}"
                       .format(index))
    graph_slice = index
  else:
    raise TypeError(
        "unsupported index type got {} with type {}. Index must be a valid "
        "scalar integer (tensor or int) or a slice of such values."
        .format(index, type(index)))

  start_slice = slice(0, graph_slice.start)

  with tf.name_scope(name):
    start_node_index = tf.reduce_sum(
        input_graphs.n_node[start_slice], name="start_node_index")
    start_edge_index = tf.reduce_sum(
        input_graphs.n_edge[start_slice], name="start_edge_index")
    end_node_index = start_node_index + tf.reduce_sum(
        input_graphs.n_node[graph_slice], name="end_node_index")
    end_edge_index = start_edge_index + tf.reduce_sum(
        input_graphs.n_edge[graph_slice], name="end_edge_index")
    nodes_slice = slice(start_node_index, end_node_index)
    edges_slice = slice(start_edge_index, end_edge_index)

    sliced_graphs_dict = {}

    for field in set(GRAPH_NUMBER_FIELDS) | {"globals"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), graph_slice)

    field = "nodes"
    sliced_graphs_dict[field] = safe_slice_none(
        getattr(input_graphs, field), nodes_slice)

    for field in {"edges", "senders", "receivers"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), edges_slice)
      if (field in {"senders", "receivers"} and
          sliced_graphs_dict[field] is not None):
        sliced_graphs_dict[field] = sliced_graphs_dict[field] - start_node_index

    return graphs.GraphsTuple(**sliced_graphs_dict)


def get_num_graphs(input_graphs, name="get_num_graphs"):
  """Returns the number of graphs (i.e. the batch size) in `input_graphs`.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing tensors.
    name: (string, optional) A name for the operation.

  Returns:
    An `int` (if a static number of graphs is defined) or a `tf.Tensor` (if the
      number of graphs is dynamic).
  """
  with tf.name_scope(name):
    return _get_shape(input_graphs.n_node)[0]


def nest_to_numpy(nest_of_tensors):
  """Converts a nest of eager tensors to a nest of numpy arrays.

  Leaves non-`tf.Tensor` elements untouched.

  A common use case for this method is to transform a `graphs.GraphsTuple` of
  tensors into a `graphs.GraphsTuple` of arrays, or nests containing
  `graphs.GraphsTuple`s.

  Args:
    nest_of_tensors: Nest containing `tf.Tensor`s.

  Returns:
    A nest with the same structure where `tf.Tensor`s are replaced by numpy
    arrays and all other elements are kept the same.
  """
  return tree.map_structure(
      lambda x: x.numpy() if isinstance(x, tf.Tensor) else x,
      nest_of_tensors)


def specs_from_graphs_tuple(
    graphs_tuple_sample,
    dynamic_num_graphs=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    description_fn=tf.TensorSpec,
    ):
  """Returns the `TensorSpec` specification for a given `GraphsTuple`.

  This method is often used with `tf.function` in Tensorflow 2 to obtain
  improved speed and performance of eager code. For example:

  ```
  example_graphs_tuple = get_graphs_tuple(...)

  @tf.function(input_signature=[specs_from_graphs_tuple(example_graphs_tuple)])
  def forward_pass(graphs_tuple_input):
    graphs_tuple_output = graph_network(graphs_tuple_input)
    return graphs_tuple_output

  for i in range(num_training_steps):
    input = get_graphs_tuple(...)
    with tf.GradientTape() as tape:
      output = forward_pass(input)
      loss = compute_loss(output)
    grads = tape.gradient(loss, graph_network.trainable_variables)
    optimizer.apply(grads, graph_network.trainable_variables)
  ```

  Args:
    graphs_tuple_sample: A `graphs.GraphsTuple` with sample data. `GraphsTuple`s
      that have fields with `None` are not accepted since they will create an
      invalid signature specification for `tf.function`. If your graph has
      `None`s use `utils_tf.set_zero_edge_features`,
      `utils_tf.set_zero_node_features` or `utils_tf.set_zero_global_features`.
      This method also returns the signature for `GraphTuple`s with nests of
      tensors in the feature fields (`nodes`, `edges`, `globals`), including
      empty nests (e.g. empty list, dict, or tuple). Nested node, edge and
      global feature tensors, should usually have the same leading dimension as
      all other node, edge and global feature tensors respectively.
    dynamic_num_graphs: Boolean indicating if the number of graphs in each
      `GraphsTuple` will be variable across examples.
    dynamic_num_nodes: Boolean indicating if number of nodes per graph will be
      variable across examples. Not used if `dynamic_num_graphs` is True, as the
      size of the first axis of all `GraphsTuple` fields will be variable, due
      to the variable number of graphs.
    dynamic_num_edges: Boolean indicating if number of edges per graph will be
      variable across examples. Not used if dynamic_num_graphs is True, as the
      size of the first axis of all `GraphsTuple` fields will be variable, due
      to the variable number of graphs.
    description_fn: A callable which accepts the dtype and shape arguments to
      describe the shapes and types of tensors. By default uses `tf.TensorSpec`.

  Returns:
    A `GraphsTuple` with tensors replaced by `TensorSpec` with shape and dtype
    of the field contents.

  Raises:
    ValueError: If a `GraphsTuple` has a field with `None`.
  """
  graphs_tuple_description_fields = {}
  edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]

  # Method to get the spec for a single tensor.
  def get_tensor_spec(tensor, field_name):
    """Returns the spec of an array or a tensor in the field of a graph."""

    shape = list(tensor.shape)
    dtype = tensor.dtype

    # If the field is not None but has no field shape (i.e. it is a constant)
    # then we consider this to be a replaced `None`.
    # If dynamic_num_graphs, then all fields have a None first dimension.
    # If dynamic_num_nodes, then the "nodes" field needs None first dimension.
    # If dynamic_num_edges, then the "edges", "senders" and "receivers" need
    # a None first dimension.
    if (shape and (
        dynamic_num_graphs or
        (dynamic_num_nodes and field_name == graphs.NODES) or
        (dynamic_num_edges and field_name in edge_dim_fields))):
      shape[0] = None
    return description_fn(shape=shape, dtype=dtype)

  for field_name in graphs.ALL_FIELDS:
    field_sample = getattr(graphs_tuple_sample, field_name)
    if field_sample is None:
      raise ValueError(
          "The `GraphsTuple` field `{}` was `None`. All fields of the "
          "`GraphsTuple` must be specified to create valid signatures that"
          "work with `tf.function`. This can be achieved with `input_graph = "
          "utils_tf.set_zero_{{node,edge,global}}_features(input_graph, 0)`"
          "to replace None's by empty features in your graph. Alternatively"
          "`None`s can be replaced by empty lists by doing `input_graph = "
          "input_graph.replace({{nodes,edges,globals}}=[]). To ensure "
          "correct execution of the program, it is recommended to restore "
          "the None's once inside of the `tf.function` by doing "
          "`input_graph = input_graph.replace({{nodes,edges,globals}}=None)"
          "".format(field_name))

    if field_name in graphs.GRAPH_FEATURE_FIELDS:
      field_spec = tree.map_structure(
          functools.partial(get_tensor_spec, field_name=field_name),
          field_sample)
    else:
      field_spec = get_tensor_spec(field_sample, field_name=field_name)

    graphs_tuple_description_fields[field_name] = field_spec

  return graphs.GraphsTuple(**graphs_tuple_description_fields)


# Convenience data container for carrying around padding.
GraphsTupleSize = collections.namedtuple(
    "GraphsTupleSize", ["num_nodes", "num_edges", "num_graphs"])

# Mapping indicating the leading size of `GraphsTuple` fields according to the
# number of nodes/edges/graphs in the `GraphsTuple`.
_GRAPH_ATTRIBUTE_TO_SIZE_MAP = {
    graphs.NODES: "num_nodes",
    graphs.EDGES: "num_edges",
    graphs.RECEIVERS: "num_edges",
    graphs.SENDERS: "num_edges",
    graphs.GLOBALS: "num_graphs",
    graphs.N_NODE: "num_graphs",
    graphs.N_EDGE: "num_graphs"
}


def _get_field_size_from_size_tuple(size_tuple, graphs_field_name):
  field_size_name = _GRAPH_ATTRIBUTE_TO_SIZE_MAP[graphs_field_name]
  return getattr(size_tuple, field_size_name)


def _assert_if_space_for_first_padding_graph(
    graphs_tuple, graphs_tuple_padded_sizes):
  """Checks if a given graph can fit in the provided padded shape.

  Args:
    graphs_tuple: A `graphs.GraphsTuple` that is checked for size.
    graphs_tuple_padded_sizes: A `GraphsTupleSize` with the sized to pad to.

  Returns:
    An assertion op indicating whether there is space for the padding graph.
  """

  # Padding graph needs to have at least one graph, and at least one node,
  # but should not need extra edges, so the number of padded nodes and graphs
  # needs to be strictly larger, than the input sizes, but it is ok if the
  # number of padded edges are equal to the number of input edges.
  graphs_tuple_sizes = get_graphs_tuple_size(graphs_tuple)

  all_fields_fit = [
      tf.less_equal(graphs_tuple_sizes.num_edges,
                    graphs_tuple_padded_sizes.num_edges),
      tf.less(graphs_tuple_sizes.num_nodes,
              graphs_tuple_padded_sizes.num_nodes),
      tf.less(graphs_tuple_sizes.num_graphs,
              graphs_tuple_padded_sizes.num_graphs),
  ]
  all_fields_fit = functools.reduce(tf.math.logical_and, all_fields_fit)

  return tf.Assert(all_fields_fit, [
      "There is not enough space to pad the GraphsTuple "
      " with sizes (#nodes, #edges, #graphs):", graphs_tuple_sizes,
      " to padded sizes of :", graphs_tuple_padded_sizes,
      "`pad_edges_to` must be larger or equal to the maximum number of edges "
      "in any `GraphsTuple` and `pad_nodes_to`/`pad_graphs_to` must be "
      "strictly larger than the maximum number of nodes/graphs in any "
      "`GraphsTuple`."
  ])


def get_graphs_tuple_size(graphs_tuple):
  """Calculates the total nodes, edges and graphs in a graph batch.

  Args:
    graphs_tuple: A `GraphsTuple`.

  Returns:
    A `GraphsTupleSizes` object containing the total number of nodes, edges and
    graphs in the `GraphsTuple`. Each value is a scalar integer `tf.Tensor`.

  """
  num_nodes = tf.reduce_sum(graphs_tuple.n_node)
  num_edges = tf.reduce_sum(graphs_tuple.n_edge)
  num_graphs = tf.shape(graphs_tuple.n_node)[0]
  return GraphsTupleSize(num_nodes, num_edges, num_graphs)


def _get_required_padding_sizes(graphs_tuple, padded_size):
  """Gets the padding size, given a GraphsTuple and the total padded sizes."""
  graph_size = get_graphs_tuple_size(graphs_tuple)
  return GraphsTupleSize(*(b - c for b, c in zip(padded_size, graph_size)))


def get_mask(valid_length, full_length):
  """Returns a mask given the valid length of a vector with trailing padding.

  This is useful for masking out padded elements from a loss. For example
  ```
  input_graphs_tuple = ...
  input_graphs_tuple_size = get_graphs_tuple_size(input_graphs_tuple)
  padded_input_graphs_tuple = pad_graphs_tuple(input_graphs_tuple,
       pad_nodes_to,...)


  per_node_loss # After graph_network computation.

  nodes_mask = get_mask(
      input_graphs_tuple_size.nodes, pad_nodes_to)

  masked_per_node_loss = per_node_loss * tf.cast(
      nodes_mask, per_node_loss.dtype)
  ```

  Args:
    valid_length: Length of the valid elements.
    full_length: Full length of the vector after padding.

  Returns:
    Boolean mask of shape [full_length], where all values are set to `True`
    except for the last `padding_length` which are set to False.
  """
  valid_length = tf.cast(valid_length, tf.int32)
  full_length = tf.cast(full_length, tf.int32)
  # This implementation allows for statically sized shapes, rather than
  # using concat([tf.ones([valid_length]), tf.zeros([full_length-valid_length])]
  # which has intermediate tensors with shapes not know statically.
  field_mask = tf.range(full_length)
  field_mask = field_mask < valid_length
  return field_mask


def remove_graphs_tuple_padding(padded_graphs_tuple, valid_size):
  """Strips a padded `GraphsTuple` of padding.

  Given a graph that has been padded by `padding` amount, remove the padding
  to recover the original graph.

  Often used in the sequence:
  ```
    graphs_tuple_size = get_graphs_tuple_size(graphs_tuple)
    padded_graphs_tuple = pad_graphs_tuple(graphs_tuple,
                                           pad_nodes_to=x,
                                           pad_edges_to=y,
                                           pad_graphs_to=z)
    unpadded_graphs_tuple = remove_graphs_tuple_padding(padded_graphs_tuple,
                                                        graphs_tuple_size)
  ```


  Args:
    padded_graphs_tuple: A `graphs.GraphsTuple` that has been padded by
      `padding` amount.
    valid_size: A `GraphsTupleSize` that represents the size of the valid graph.

  Returns:
    Returns a `graphs.GraphsTuple` which is padded_graphs_tuple stripped of
      padding.
  """
  stripped_graph_kwargs = {}
  graph_dict = padded_graphs_tuple._asdict()  # pylint: disable=protected-access

  for field, tensor_nest in graph_dict.items():
    field_valid_size = _get_field_size_from_size_tuple(valid_size, field)
    strip_fn = lambda x: x[:field_valid_size]  # pylint:disable=cell-var-from-loop
    stripped_field = tree.map_structure(strip_fn, tensor_nest)
    stripped_graph_kwargs[field] = stripped_field
  return graphs.GraphsTuple(**stripped_graph_kwargs)


def _pad_tensor(tensor, field, padding_size):
  """Pads a tensor on the first dimensions with the padding size.

  Args:
    tensor: tf.Tensor of size [batch_dim, x1, ..., xn].
    field: Text, the field of `graphs.GraphsTuple` to pad.
    padding_size: A tuple representing the size of padding of the graph.

  Returns:
    A tf.Tensor of size [batch_dim + padding, x1, ..., xn] padded with zeros.
  """
  padding = _get_field_size_from_size_tuple(padding_size, field)
  padding_tensor = tf.zeros(
      [padding] + tensor.shape.as_list()[1:],
      dtype=tensor.dtype,
      name="pad_zeros_{}".format(field))
  return tf.concat((tensor, padding_tensor), axis=0)


def _get_zeros_with_variable_batch_size(feature_tensor, padding_size):
  return tf.zeros([padding_size] + feature_tensor.shape.as_list()[1:],
                  feature_tensor.dtype)


def _get_first_padding_graph(graphs_batch, padding_size,
                             experimental_unconnected_padding_edges):
  """Gets a dummy graph that pads receivers and senders.

  This dummy graph will have number of nodes = padding_size.nodes and
  number of edges = padding_size.edges. Receivers and
  senders will be indexed with all zeros (connecting to the first node in the
  dummy graph).

  Args:
    graphs_batch: the `graphs.GraphsTuple` to be padded.
    padding_size: a `GraphsTupleSize` with the padding size.
    experimental_unconnected_padding_edges: see `pad_graphs_tuple` for details.

  Returns:
    A `graphs.GraphsTuple` of a single dummy graph.
  """

  # Set the edges to belong to an index corresponding to a node that does not
  # exist.
  if experimental_unconnected_padding_edges:
    logging.log_first_n(
        logging.WARNING,
        "Using a padding graph with unconnected edges. This is an experimental "
        "feature which may stop working in the future, and will lead to out"
        "of range errors on tf.scatter if the graph net computation occurs on "
        "CPU.", 1)
    dummy_senders_and_receivers = (
        tf.ones([padding_size.num_edges], tf.int32) * padding_size.num_nodes)
  else:
    dummy_senders_and_receivers = tf.zeros([padding_size.num_edges], tf.int32)

  return graphs.GraphsTuple(
      n_node=[padding_size.num_nodes],
      n_edge=[padding_size.num_edges],
      nodes=tree.map_structure(
          functools.partial(
              _get_zeros_with_variable_batch_size,
              padding_size=padding_size.num_nodes), graphs_batch.nodes),
      edges=tree.map_structure(
          functools.partial(
              _get_zeros_with_variable_batch_size,
              padding_size=padding_size.num_edges), graphs_batch.edges),
      senders=dummy_senders_and_receivers,
      receivers=dummy_senders_and_receivers,
      globals=tree.map_structure(
          functools.partial(
              _get_zeros_with_variable_batch_size, padding_size=1),
          graphs_batch.globals))


def pad_graphs_tuple(graphs_tuple,
                     pad_nodes_to,
                     pad_edges_to,
                     pad_graphs_to,
                     experimental_unconnected_padding_edges=False):
  """Pads a `graphs.GraphsTuple` to fixed number of nodes, edges and graphs.

  The Graph Nets library treat zeros as valid parts of a graph.GraphsTuple, so
  special padding is required in order to preserve the computation. This
  method does so by adding a 'dummy' graph to the batch so that additional
  nodes/edges can't interfere with the valid graph.

  Args:
    graphs_tuple: `graphs.GraphsTuple` batch of graphs.
    pad_nodes_to: the size to pad node determined features to.
    pad_edges_to: the size to pad edge determined features to.
    pad_graphs_to: the size to pad graph determined features to.
    experimental_unconnected_padding_edges: Experimental feature to prevent nans
      in the padding graph. DISCLAIMER: This feature is extremly experimental,
      and setting it to `True` is not recommened unless strictly necessary, and
      understanding the implications.

      If `True`, the padding graph will have `senders` and `receivers` for
      the padding edges reference a node which does not exist (one beyond the
      size of `nodes`).

      This feature can be used to prevent any broadcasting/aggregation ops
      between edges and nodes for the padding graph. The reason is that the
      sum aggregations in the padding graph, which has a single node with a
      very large number of self-edges, sometimes lead to infs or nans,
      which may contaminate the gradients of the other valid graphs in the batch
      with nans (even if masked out of the loss: this is related to the
      `tf.where` issue.).

      This approach relies on several numerical hacks that do not work on CPU,
      but work on GPU and TPU (as covered by our tests):

      * `tf.gather` returns zeros when the index is beyond the boundaries. From
         https://www.tensorflow.org/api_docs/python/tf/gather
           "Note that on CPU, if an out of bound index is found, an error is
           returned. On GPU, if an out of bound index is found, a 0 is stored
           in the corresponding output value."
      * `tf.unsorted_segment_sum` drops values for negative indices. From
         https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum
           "If the given segment ID is negative, the value is dropped and
           will not be added to the sum of the segment."
        We have seen empirically that it also ignores values with indices equal
        or larger than `num_segments`. While this behavior is tested in our
        library, we cannot guarantee that it will work in the future for all
        unsorted_segment ops, so use at your own risk.

      This fixes the appearance of nans in the node-wise edge aggregation. The
      appearance of `nan`s is less likely in the global aggregation because in
      the worst case, the number of nodes/edges on the padding graph is not
      typically much larger than the number of nodes/edges in other graphs in
      the dataset.

      A less hacky approach (but more expensive, and requiring modifying model
      code) to prevent nan's appearing in the padding graph, is by masking out
      the graph features before they are aggregated, although for convenience
      we usually find that it is enough to do it after each message passing
      layer. E.g.:

      ```
      graphs_tuple_size = get_graphs_tuple_size(graphs_tuple)

      padded_graphs_tuple = pad_graphs_tuple(graphs_tuple, ...)

      graphs_mask = get_mask(graphs_tuple_size.num_graphs, pad_graphs_to)
      nodes_mask = get_mask(graphs_tuple_size.num_nodes, pad_nodes_to)
      edges_mask = get_mask(graphs_tuple_size.num_edges, pad_edges_to)

      # Some computation that creates intermediate `any_padded_graphs_tuple`s
      # after each message passing step.
      any_padded_graphs_tuple = any_padded_graphs_tuple.replace(
          edges=any_padded_graphs_tuple.edges * tf.cast(
              edges_mask, tf.float32)[:, None],
          nodes=any_padded_graphs_tuple.nodes * tf.cast(
              nodes_mask, tf.float32)[:, None],
          globals=any_padded_graphs_tuple.globals * tf.cast(
              graphs_mask, tf.float32)[:, None],
      )
      ```

  Returns:
    A `graphs.GraphsTuple` padded up to the required values.
  """
  padded_sizes = GraphsTupleSize(pad_nodes_to, pad_edges_to, pad_graphs_to)

  # The strategy goes as follows:
  # 0. Make sure our `graphs_tuple` is at least 1 node and 1 graph smaller than
  #    the padded sizes (this is required for step 1).
  # 1. Pad with one graph with at least one node, that contains all padding
  #    edges, and padding nodes, this will guaranteed preserved computation for
  #    graphs in the input `GraphsTuple`.
  # 2. Pad up to `pad_graphs_to` with graphs with no nodes and no edges.
  # 3. Set the shapes of the padded tensors to be statically known. Otherwise
  #    tensorflow shape inference mechanism is not smart enough to realize that
  #    at this stage tensors have statically known sizes.

  # Step 0.
  sufficient_space_assert = _assert_if_space_for_first_padding_graph(
      graphs_tuple, padded_sizes)
  with tf.control_dependencies([sufficient_space_assert]):
    padding_size = _get_required_padding_sizes(graphs_tuple, padded_sizes)

  # Step 1.
  first_padding_graph = _get_first_padding_graph(
      graphs_tuple, padding_size, experimental_unconnected_padding_edges)
  graphs_tuple_with_first_padding_graph = concat(
      [graphs_tuple, first_padding_graph], axis=0)

  # Step 2.
  remaining_padding_sizes = _get_required_padding_sizes(
      graphs_tuple_with_first_padding_graph, padded_sizes)
  padded_batch_kwargs = {}
  for field, tensor_dict in (
      graphs_tuple_with_first_padding_graph._asdict().items()):  # pylint: disable=protected-access
    field_pad_fn = functools.partial(
        _pad_tensor, padding_size=remaining_padding_sizes, field=field)
    padded_batch_kwargs[field] = tree.map_structure(field_pad_fn, tensor_dict)

  # Step 3.
  def _set_shape(tensor, padded_size):
    tensor_shape = tensor.get_shape().as_list()
    tensor.set_shape([padded_size] + tensor_shape[1:])
    return tensor
  for field, tensor_dict in padded_batch_kwargs.items():
    padded_size = _get_field_size_from_size_tuple(padded_sizes, field)
    set_shape_partial = functools.partial(_set_shape, padded_size=padded_size)
    tensor_dict = tree.map_structure(set_shape_partial, tensor_dict)
    padded_batch_kwargs[field] = tensor_dict
  return graphs.GraphsTuple(**padded_batch_kwargs)

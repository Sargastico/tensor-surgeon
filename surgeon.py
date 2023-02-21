from tensorflow import keras
from keras import layers as KL
from keras import Model

"""
Fully based on: 
- Repo: https://github.com/nrasadi/split-keras-tensorflow-model | 
- Email: na10web@gmail.com
"""

def split_network(model: keras.Model, split_layer_name: str, on_head=True, names=('head', 'tail'),
                  custom_objects: dict = None) -> (
        keras.Model, keras.Model):
    """
    Splits the neural network from the point named 'split_layer_name' into two parts.
    Arguments:
     model: Keras loaded model or path to saved model. In the case of path, the type must be 'str' or 'pathlib.Path'.
     split_layer_name: The name of the layer to be splitted.
     on_head (optional): If True, the splitted layer would belong to the head part. Otherwise it would be part of the tail model.
     names (optional): A tuple of two strings containing name of head and tail models.
     custom_objects: If the model you want to split includes custom layers or other custom classes or functions,
                     you have to pass them as a dictionary of (key:"layer/object name", value:layer/object) via this argument.
                     e.g.: custom_objects={'AttentionLayer': AttentionLayer}
    Returns:
     tuple(head_model, tail_model)
    """

    # Determine the split point based on the 'on_head' argument.
    if on_head:
        split_layer_name = model.get_layer(name=split_layer_name).outbound_nodes[0].layer.name

    tail_input = KL.Input(
        batch_shape=model.get_layer(split_layer_name).get_input_shape_at(0))

    layer_outputs = {}

    def _find_backwards(layer):
        """
        Returns outputs of a layer by moving backward and
        finding outputs of previous layers until reaching split layer.
        directly inspired by the answer at the link below
        with some modifications and corrections. https://stackoverflow.com/a/56228514
        This is an internal function.
        """

        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        if layer.name == split_layer_name:
            out = layer(tail_input)
            layer_outputs[layer.name] = out
            return out

        # Find all the connected layers which this layer consumes their output
        prev_layers = []
        for node in layer.inbound_nodes:

            try:
                # If number of inbound layers > 1
                prev_layers.extend(node.inbound_layers)
            except TypeError:
                # If number of inbound layers == 1
                prev_layers.append(node.inbound_layers)

        # Get the output of connected layers in a recursive manner
        pl_outs = []
        for pl in prev_layers:
            plo = _find_backwards(pl)
            try:
                pl_outs.extend([plo])
            except TypeError:
                pl_outs.append([plo])

        # Apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out

        return out

    tail_output = _find_backwards(model.layers[-1])

    # Creating head and tail models
    head_model = Model(model.input, model.get_layer(split_layer_name).output, name=names[0])
    tail_model = Model(tail_input, tail_output, name=names[1])

    return head_model, tail_model

from SADNet.model.sadnet import SADNET


def make_model(input_channel, output_channel):
    return SADNET(input_channel, output_channel, 32, 32)

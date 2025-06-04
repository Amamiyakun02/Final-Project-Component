def gather_params(model):
    params = []
    # contoh untuk Conv2D
    params.append(model.conv1.W)
    params.append(model.conv1.b)
    # loop semua ResidualBlock dan ambil paramnya
    for block in model.layer1 + model.layer2 + model.layer3 + model.layer4:
        params.extend([block.conv1.W, block.conv1.b,
                       block.conv2.W, block.conv2.b])
        if block.use_projection:
            params.extend([block.shortcut_conv.W, block.shortcut_conv.b])
    # Dense layer
    params.append(model.fc.W)
    params.append(model.fc.b)
    return params

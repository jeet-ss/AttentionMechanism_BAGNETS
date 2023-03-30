import torch
import bagnets.pytorchnet

# only download and save the model

if __name__ == '__main__':

    #
    bagnets_model = bagnets.pytorchnet.bagnet17(avg_pool=True, pretrained=True)

    #
    print("Model's state_dict:")
    for param_tensor in bagnets_model.state_dict():
        print(param_tensor, "\t", bagnets_model.state_dict()[param_tensor].size())

    #
    model_path = "/home/woody/iwso/iwso060h/Model/bagnet17_trained.pt"
    torch.save(bagnets_model, model_path)


from agentDetection import AutoDict
from torchvision.models import vgg


def vggFeature(inp_frames):
    vgg16 = vgg.vgg16(pretrained=True).cuda()
    output = vgg16.features[:4](inp_frames)
    # print(output.shape)
    # print(br)
    return output


def extract_Features(inp,agent_tubes):
    ### To be updated for parallel processing
    ### To be updated for loading once
    agent_feat = []
    for batch in range(inp.shape[0]):
        print(batch)
        feats = vggFeature(inp[batch])
        agent_feat.append(feats)
    return agent_feat


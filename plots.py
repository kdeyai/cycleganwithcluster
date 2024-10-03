import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn
import matplotlib

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True).cuda()
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

l = []
target = []
count = 0
for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainA/'):
    im = Image.open('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainA/'+i)
    img = transform(im).view(1, 3, 256, 256).cuda()
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    output = model(img)
    l.append(activation['avgpool'].squeeze())
    target.append(0)
    count+=1
    # if count > 501:
    #     break;

# m = torch.stack(l).cpu().numpy()
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(m)

# l = []
# target = []
count1 = 0

for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'):
    
    im = Image.open('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
    img = transform(im).view(1, 3, 256, 256).cuda()
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    output = model(img)
    l.append(activation['avgpool'].squeeze())  
    target.append(1)
    count1+=1
    # if count1> 501:
    #     break;


# m = torch.stack(l).cpu().numpy()
# X_embedded1 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(m)      

# l = []
# target = []
count2 = 0
for i in os.listdir('/home/ishika/kaushik/art2real/resultscluster/landscape2photo/test_latest/images/'):
        if i.endswith('_fake.png'):
            im = Image.open('/home/ishika/kaushik/art2real/resultscluster/landscape2photo/test_latest/images/'+i)
            img = transform(im).view(1, 3, 256, 256).cuda()
            model.avgpool.register_forward_hook(get_activation('avgpool'))
            output = model(img)
            l.append(activation['avgpool'].squeeze())  
            target.append(2)  
            count2+=1

m = torch.stack(l).cpu().numpy()
print(m.shape)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(m)


print(count,count1, count2)
matplotlib.rcParams['lines.markersize'] = 3

fig, ax = plt.subplots(1,1)
plt.scatter(X_embedded[0: count,0], X_embedded[0: count,1], label = 'paint')
plt.scatter(X_embedded[count+1:count+count1,0], X_embedded[count+1:count+count1,1], label = 'real')
plt.scatter(X_embedded[count+count1+1:count+count1+count2,0], X_embedded[count+count1+1:count+count1+count2,1], label = 'generated')
plt.legend()
plt.savefig('cyclegancluster.png')

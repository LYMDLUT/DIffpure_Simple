import os
import numpy
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline, DDPMScheduler
import torch
import os
import diffusers
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers.utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from PIL import Image

Denoise_step = 10
class DDPMPipeline_Img2Img(DDPMPipeline):
    #@torch.no_grad()
    def __call__(
        self,
        sample_image,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # Sample gaussian noise to begin loop
        # if isinstance(self.unet.config.sample_size, int):
        #     image_shape = (
        #         batch_size,
        #         self.unet.config.in_channels,
        #         self.unet.config.sample_size,
        #         self.unet.config.sample_size,
        #     )
        # else:
        #     image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        #
        # if self.device.type == "mps":
        #     # randn does not work reproducibly on mps
        #     image = randn_tensor(image_shape, generator=generator)
        #     image = image.to(self.device)
        # else:
        #     image = randn_tensor(image_shape, generator=generator, device=self.device)
        image = sample_image
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        #aa = torch.autograd.grad(image.mean(), sample_image)
        for t in self.progress_bar(reversed(range(Denoise_step))):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image1 = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image1 = self.numpy_to_pil(image1)
        #
        # if not return_dict:
        #     return (image,)
       # image.mean().backward()
        return image, image1


model_id = "google/ddpm-cifar10-32"
batch_size = 10
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
mu = torch.tensor(mean).view(3, 1, 1)
std1 = torch.tensor(std).view(3, 1, 1)
upper_limit = ((1 - mu)/ std1)
lower_limit = ((0 - mu)/ std1)
transform_cifar10 = transforms.Compose(
    [transforms.Normalize(mean, std)]
)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
cifar10_test_loader = DataLoader(
    cifar10_test, shuffle=False, num_workers=0, batch_size=batch_size)

states_att = torch.load('origin.t7', map_location='cpu')  # Temporary t7 setting
network_clf = states_att['net'].to('cpu')
network_clf.eval()

noise_schdeuler = DDPMScheduler(num_train_timesteps=1000)
ddpm = DDPMPipeline_Img2Img.from_pretrained(model_id)

sample_image, y_val = next(iter(cifar10_test_loader))
timesteps = torch.LongTensor([Denoise_step])



epsilon = (8 / 255.) / std1
alpha = (2 / 255.) / std1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

delta = torch.zeros_like(sample_image)
delta.requires_grad = True
for _ in range(10):
    eot = 0
    for _ in range(1):
        noise = torch.randn(sample_image.shape)
        noisy_image = noise_schdeuler.add_noise(sample_image + delta, noise, timesteps)
        #noisy_image.mean().backward()
        #noisy_image.retain_grad()
        images_1, images_2 = ddpm(sample_image=noisy_image, batch_size=noisy_image.shape[0], num_inference_steps=1000)
        # images_1.requires_grad = True
        # images_1.retain_grad()
        tmp_in = transform_cifar10(images_1)
        tmp_out = network_clf(tmp_in)
        loss = F.cross_entropy(tmp_out, y_val)
        loss.backward()
        #print(noisy_image.grad)
        grad = delta.grad.detach()
        eot += grad
        delta.grad.zero_()
    d = clamp(delta + alpha * eot.sign(), -epsilon, epsilon)
    d = clamp(d, lower_limit - sample_image, upper_limit - sample_image)
    delta.data = d
    #delta.grad.zero_()


adv_out = (sample_image + delta)
noise = torch.randn(sample_image.shape)
noisy_image = noise_schdeuler.add_noise(adv_out, noise, timesteps)
images_1, images_2 = ddpm(sample_image=noisy_image, batch_size=noisy_image.shape[0], num_inference_steps=1000)


plt.figure()
plt.imshow(((sample_image[0]/ 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8), vmin=0, vmax=255)
plt.show()

plt.figure()
plt.imshow(((adv_out[0]/2+0.5).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8), vmin=0, vmax=255)
plt.show()
# #
plt.figure()
plt.imshow(((noisy_image[0]/ 2 + 0.5).permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8), vmin=0, vmax=255)
plt.show()

# run pipeline in inference (sample random noise and denoise)

#img = Image.open('ddpm_generated_image.png')
#plt.figure()
#img = Image.fromarray((np.array(image) * 255.0).astype(np.uint8)).convert("BGR")

#images_2[0].show()
plt.imshow(((np.array(images_2[0])).astype(np.uint8)), vmin=0, vmax=255)
plt.show()

with torch.no_grad():
    yy = network_clf(transform_cifar10(images_1))
    print(sum((yy.argmax(dim=-1).cpu() == y_val)) / y_val.shape[0])
    #print(sum((yy1.argmax(dim=-1).cpu() == y_val)) / y_val.shape[0])

# save image
#image.save("ddpm_generated_image.png")

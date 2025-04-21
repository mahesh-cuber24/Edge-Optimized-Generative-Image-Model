import torch
import torch.nn.utils.prune as prune
from models.gan import Generator

model = Generator()
model.load_state_dict(torch.load("generator.pth"))
model.eval()

# Apply pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.ConvTranspose2d):
        prune.l1_unstructured(module, name="weight", amount=0.4)

# Convert to a quantized model (dummy example)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

torch.save(model.state_dict(), "generator_pruned_quantized.pth")

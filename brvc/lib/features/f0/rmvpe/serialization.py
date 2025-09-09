import torch
from .e2e import E2E
from safetensors.torch import load_file, save_file


def load_rmvpe(
    model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu"), is_half=False
) -> E2E:
    """
    """
    model = E2E(4, 1, (2, 2))

    if model_path.endswith(".pt"):
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    elif model_path.endswith(".safetensors"):
        ckpt = load_file(model_path, device=device.type)
    else:
        raise TypeError(f"Unsupported model path: {model_path}")
    model.load_state_dict(ckpt)
    del ckpt
    model.eval()
    if is_half:
        model = model.half()
    model = model.to(device)
    return model

def convert_pt_to_safetensors(
    pt_path="assets/rmvpe/rmvpe.pt",
    safetensors_path="assets/rmvpe/rmvpe.safetensors",
):
    """
    Converts a .pt model checkpoint to a .safetensors file.
    """
    try:
        print(f"Loading weights from: {pt_path}")
        # Load the weights from the .pt file
        pt_weights = torch.load(pt_path, map_location="cpu", weights_only=True)

        print(f"Saving weights to: {safetensors_path}")
        # Save the weights to a .safetensors file
        save_file(pt_weights, safetensors_path)

        print("Conversion successful!")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")


def test():
    # Example usage:
    # This part will only run if you execute the script directly.
    # Make sure to have your model files in the specified paths
    # or update the paths accordingly.

    # 1. Define the paths
    original_pt_path = "assets/rmvpe/rmvpe.pt"
    new_safetensors_path = "assets/rmvpe/rmvpe.safetensors"

    # 2. Convert the .pt file to .safetensors
    # Note: You would need to run this once.
    # convert_pt_to_safetensors(original_pt_path, new_safetensors_path)

    # 3. Now you can load the model from either file
    print("Attempting to load model from .pt file...")
    try:
        # You need a dummy e2e.py file for this to run
        model_from_pt = load_rmvpe(model_path=original_pt_path)
        print("Successfully loaded model from .pt")
        # print(model_from_pt)
    except FileNotFoundError:
        print(f"File not found: {original_pt_path}. Skipping .pt loading example.")
    except ImportError:
        print("Could not import E2E model. Skipping example.")

    print("\nAttempting to load model from .safetensors file...")
    try:
        # You need a dummy e2e.py file for this to run
        model_from_sf = load_rmvpe(model_path=new_safetensors_path)
        print("Successfully loaded model from .safetensors")
        # print(model_from_sf)
    except FileNotFoundError:
        print(f"File not found: {new_safetensors_path}. Make sure to convert it first.")
    except ImportError:
        print("Could not import E2E model. Skipping example.")

if __name__ == "__main__":
    test()
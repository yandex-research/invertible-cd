import numpy as np
import torch


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def predicted_origin(
    model_output, timesteps, boundary_timesteps, sample, prediction_type, alphas, sigmas
):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    # Set hard boundaries to ensure equivalence with reverse (forward) CD
    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas  # x0 prediction
        pred_x_0 = (
            alphas_s * pred_x_0 + sigmas_s * model_output
        )  # Euler step to the boundary step
    elif prediction_type == "v_prediction":
        assert (
            boundary_timesteps == 0
        ), "v_prediction does not support multiple endpoints at the moment"
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(
        self,
        alpha_cumprods,
        timesteps=1000,
        ddim_timesteps=50,
        num_endpoints=1,
        num_forward_endpoints=1,
        endpoints=None,
        forward_endpoints=None,
    ):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1  # [19, ..., 999]
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_alpha_cumprods_next = np.asarray(
            alpha_cumprods[self.ddim_timesteps[1:]].tolist() + [0.0]
        )
        # Convert to torch.tensor
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
        self.ddim_alpha_cumprods_next = torch.from_numpy(self.ddim_alpha_cumprods_next)

        # Set boundary time steps for reverse CD
        if endpoints is None:
            timestep_interval = ddim_timesteps // num_endpoints + int(
                ddim_timesteps % num_endpoints > 0
            )
            endpoint_idxs = (
                torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
            )
            self.endpoints = torch.tensor(
                [0] + self.ddim_timesteps[endpoint_idxs].tolist()
            )
        else:
            self.endpoints = torch.tensor(
                [int(endpoint) for endpoint in endpoints.split(",")]
            )
            assert len(self.endpoints) == num_endpoints

        # Set boundary time steps for forward CD
        if forward_endpoints is None:
            timestep_interval = ddim_timesteps // num_forward_endpoints + int(
                ddim_timesteps % num_forward_endpoints > 0
            )
            forward_endpoint_idxs = (
                torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
            )
            forward_endpoint_idxs = torch.tensor(
                forward_endpoint_idxs.tolist() + [ddim_timesteps - 1]
            )
            self.forward_endpoints = self.ddim_timesteps[forward_endpoint_idxs]
        else:
            self.forward_endpoints = torch.tensor(
                [int(endpoint) for endpoint in forward_endpoints.split(",")]
            )
            assert len(self.forward_endpoints) == num_forward_endpoints

        print(f"Boundary timesteps reverse CD: {self.endpoints} | Boundary timesteps forward CD: {self.forward_endpoints}")

    def to(self, device):
        self.endpoints = self.endpoints.to(device)
        self.forward_endpoints = self.forward_endpoints.to(device)

        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        self.ddim_alpha_cumprods_next = self.ddim_alpha_cumprods_next.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def forward_ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_next = extract_into_tensor(
            self.ddim_alpha_cumprods_next, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_next).sqrt() * pred_noise
        x_next = alpha_cumprod_next.sqrt() * pred_x0 + dir_xt
        return x_next

from abc import abstractmethod, ABC
import torch


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """
    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(
        self, clean_latent: torch.Tensor,
        noise: torch.Tensor, timestep: torch.Tensor
    ):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        pass

    @abstractmethod
    def add_noise_high(
        self, clean_latent, noise, timestep, timestep_start
    ):
        pass

    @abstractmethod
    def add_noise_low(
        self, clean_latent, noise, timestep, timestep_start
    ):
        pass

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(
            lambda x: x.double().to(x0.device), [x0, xt,
                                                 self.alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t **
                      (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(
            lambda x: x.double().to(noise.device), [noise, xt,
                                                    self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t **
                   (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, velocity: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device), [velocity, xt,
                                                       self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t ** 0.5) * xt - (beta_prod_t ** 0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep_id):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        self.sigmas = self.sigmas.to(noise.device)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    @staticmethod
    def calculate_alpha_beta_high(sigma, sigma_bound):
        alpha = (1 - sigma) / (1 - sigma_bound)
        beta = torch.sqrt(sigma ** 2 - (alpha * sigma_bound) ** 2)
        return alpha, beta

    @staticmethod
    def calculate_alpha_beta_low(sigma, sigma_bound):
        beta = sigma / sigma_bound
        alpha = 1 - beta
        return alpha, beta
    
    def add_noise_high(self, original_samples, noise, timestep_id, timestep_bound):
        timestep_bound = timestep_bound.to(self.timesteps.device)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        sigma_t = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        timestep_id_bound = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep_bound.unsqueeze(1)).abs(), dim=1)
        sigma_t_bound = self.sigmas[timestep_id_bound].reshape(-1, 1, 1, 1)
        alpha, beta = self.calculate_alpha_beta_high(sigma_t, sigma_t_bound)
        sample = alpha * original_samples + beta * noise
        return sample.type_as(noise)

    def add_noise_low(self, original_samples, noise, timestep_id, timestep_bound):
        self.sigmas = self.sigmas.to(noise.device)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


if __name__ == '__main__':
    scheduler = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True
    )
    scheduler.set_timesteps(1000)

    target_timesteps = torch.tensor([1000, 875, 750, 625, 500, 375, 250, 125])

    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    denoising_step_list = timesteps[1000 - target_timesteps]
    print(f"denoising_step_list: {denoising_step_list}")

    target_timesteps = 5.0 * (target_timesteps / 1000) / \
                    (1 + (5.0 - 1) * (target_timesteps / 1000)) * 1000

    print("target_timesteps:")
    print(target_timesteps)

    for step in target_timesteps:
        step = step.item()
        timestep_id = torch.argmin(
            (scheduler.timesteps.unsqueeze(0) - torch.tensor([step]).unsqueeze(1)).abs(), dim=1)
        print("================================================")
        print(f"step: {step}, timestep_id: {timestep_id}")
        print(f"sigma: {scheduler.sigmas[timestep_id]}")
        print(f"timestep: {scheduler.timesteps[timestep_id]}")


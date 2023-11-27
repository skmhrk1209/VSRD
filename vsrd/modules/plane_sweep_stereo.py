import torch
import torch.nn as nn

from .. import operations


class PlaneSweepStereo(nn.Module):

    def __init__(
        self,
        depth_range,
        num_hypotheses,
        depth_to_disparity,
        disparity_to_depth,
        cost_function,
        interp_mode="bilinear",
        padding_mode="zeros",
    ):
        super().__init__()
        self.depth_range = depth_range
        self.num_hypotheses = num_hypotheses
        self.depth_to_disparity = depth_to_disparity
        self.disparity_to_depth = disparity_to_depth
        self.cost_function = cost_function
        self.interp_mode = interp_mode
        self.padding_mode = padding_mode

    def forward(self, target_inputs, *multi_source_inputs, depth_range=None):

        depth_range = torch.as_tensor(self.depth_range if depth_range is None else depth_range)
        disparity_range = torch.sort(self.depth_to_disparity(depth_range)).values

        hypothesis_disparities = torch.linspace(*disparity_range, self.num_hypotheses).to(target_inputs.feature_maps)
        hypothesis_disparity_volumes = hypothesis_disparities.reshape(1, 1, -1, 1, 1).expand(-1, -1, -1, *target_inputs.feature_maps.shape[-2:])
        hypothesis_depth_volumes = self.disparity_to_depth(hypothesis_disparity_volumes)

        target_cost_volumes = torch.mean(torch.stack([
            torch.stack([
                torch.mean(self.cost_function(
                    operations.backward_warping(
                        source_feature_maps=source_inputs.feature_maps,
                        target_depth_maps=hypothesis_depth_maps,
                        source_intrinsic_matrices=source_inputs.intrinsic_matrices.new_tensor([[
                            [0.5 ** source_inputs.scale, 0.0, 0.0],
                            [0.0, 0.5 ** source_inputs.scale, 0.0],
                            [0.0, 0.0, 1.0],
                        ]]) @ source_inputs.intrinsic_matrices,
                        target_intrinsic_matrices=target_inputs.intrinsic_matrices.new_tensor([[
                            [0.5 ** target_inputs.scale, 0.0, 0.0],
                            [0.0, 0.5 ** target_inputs.scale, 0.0],
                            [0.0, 0.0, 1.0],
                        ]]) @ target_inputs.intrinsic_matrices,
                        source_extrinsic_matrices=source_inputs.extrinsic_matrices,
                        target_extrinsic_matrices=target_inputs.extrinsic_matrices,
                        interp_mode=self.interp_mode,
                        padding_mode=self.padding_mode,
                        align_corners=False,
                    ),
                    target_inputs.feature_maps,
                ), dim=1)
                for hypothesis_depth_maps in torch.unbind(hypothesis_depth_volumes, dim=-3)
            ], dim=1)
            for source_inputs in multi_source_inputs
        ], dim=0), dim=0)

        return target_cost_volumes

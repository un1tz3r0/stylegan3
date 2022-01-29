import os
from typing import List, Optional, Union, Tuple
import click

import dnnlib
from torch_utils import gen_utils

import scipy
import numpy as np
import PIL.Image
import torch

import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor


# ----------------------------------------------------------------------------


# We group the different types of generation (images, grid, video, wacky stuff) into a main function
@click.group()
def main():
		pass


# ----------------------------------------------------------------------------


@main.command(name='images')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Recreate snapshot grid during training (doesn't work!!!)
@click.option('--recreate-snapshot-grid', 'training_snapshot', is_flag=True, help='Add flag if you wish to recreate the snapshot grid created during training')
@click.option('--snapshot-size', type=click.Choice(['1080p', '4k', '8k']), help='Size of the snapshot', default='4k', show_default=True)
# Synthesis options (feed a list of seeds or give the projected w to synthesize)
@click.option('--seeds', type=gen_utils.num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=gen_utils.num_range, help='List of class labels (unconditional if not specified)', default=None)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file; can be either .npy or .npz files', type=click.Path(exists=True, dir_okay=False), metavar='FILE')
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
# Grid options
@click.option('--save-grid', help='Use flag to save image grid', is_flag=True, show_default=True)
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Grid width (number of columns)', default=None)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Grid height (number of rows)', default=None)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'grid'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='generate-images', show_default=True)
def generate_images(
				ctx: click.Context,
				network_pkl: str,
				training_snapshot: bool,
				snapshot_size: str,
				seeds: Optional[List[int]],
				truncation_psi: float,
				class_idx: Optional[List[int]],
				noise_mode: str,
				projected_w: Optional[Union[str, os.PathLike]],
				new_center: Tuple[str, Union[int, np.ndarray]],	 # TODO
				save_grid: bool,
				grid_width: int,
				grid_height: int,
				outdir: str,
				description: str,
):
		"""Generate images using pretrained network pickle.

		Examples:

		\b
		# Generate curated MetFaces images without truncation (Fig.10 left)
		python generate.py generate-images --trunc=1 --seeds=85,265,297,849 \\
				--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

		\b
		# Generate uncurated MetFaces images with truncation (Fig.12 upper left)
		python generate.py generate-images --trunc=0.7 --seeds=600-605 \\
				--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

		\b
		# Generate class conditional CIFAR-10 images (Fig.17 left, Car)
		python generate.py generate-images --seeds=0-35 --class=1 \\
				--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

		\b
		# Render an image from projected W
		python generate.py generate-images --projected_w=projected_w.npz \\
				--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
		"""
		print(f'Loading networks from "{network_pkl}"...')
		device = torch.device('cuda')
		with dnnlib.util.open_url(network_pkl) as f:
				G = legacy.load_network_pkl(f)['G_ema'].to(device)	# type: ignore

		description = 'generate-images' if len(description) == 0 else description
		# Create the run dir with the given name description
		run_dir = gen_utils.make_run_dir(outdir, description)

		# Synthesize the result of a W projection.
		if projected_w is not None:
				if seeds is not None:
						print('warn: --seeds is ignored when using --projected-w')
				print(f'Generating images from projected W "{projected_w}"')
				ws, ext = gen_utils.get_w_from_file(projected_w, return_ext=True)
				ws = torch.tensor(ws, device=device)
				assert ws.shape[1:] == (G.num_ws, G.w_dim)
				n_digits = int(np.log10(len(ws))) + 1	 # number of digits for naming the .jpg images
				if ext == '.npy':
						img = gen_utils.w_to_img(G, ws, noise_mode)[0]
						PIL.Image.fromarray(img, 'RGB').save(f'{run_dir}/proj.jpg')
				else:
						for idx, w in enumerate(ws):
								img = gen_utils.w_to_img(G, w, noise_mode)[0]
								PIL.Image.fromarray(img, 'RGB').save(f'{run_dir}/proj{idx:0{n_digits}d}.jpg')
				return

		# Labels.
		label = torch.zeros([1, G.c_dim], device=device)
		if G.c_dim != 0:
				if class_idx is None:
						ctx.fail('Must specify class label with --class when using a conditional network')
				label[:, class_idx] = 1
		else:
				if class_idx is not None:
						print('warn: --class=lbl ignored when running on an unconditional network')

		if training_snapshot:
				# This doesn't really work, so more work is warranted; TODO: move it to torch_utils/gen_utils.py
				print('Recreating the snapshot grid...')
				size_dict = {'1080p': (1920, 1080, 3, 2), '4k': (3840, 2160, 7, 4), '8k': (7680, 4320, 7, 4)}
				grid_width = int(np.clip(size_dict[snapshot_size][0] // G.img_resolution, size_dict[snapshot_size][2], 32))
				grid_height = int(np.clip(size_dict[snapshot_size][1] // G.img_resolution, size_dict[snapshot_size][3], 32))
				num_images = grid_width * grid_height

				rnd = np.random.RandomState(0)
				torch.manual_seed(0)
				all_indices = list(range(70000))	# irrelevant
				rnd.shuffle(all_indices)

				grid_z = rnd.randn(num_images, G.z_dim)	 # TODO: generate with torch, as in the training_loop.py file
				grid_img = gen_utils.z_to_img(G, torch.from_numpy(grid_z).to(device), label, truncation_psi, noise_mode)
				PIL.Image.fromarray(gen_utils.create_image_grid(grid_img, (grid_width, grid_height)),
														'RGB').save(os.path.join(run_dir, 'fakes.jpg'))
				print('Saving individual images...')
				for idx, z in enumerate(grid_z):
						z = torch.from_numpy(z).unsqueeze(0).to(device)
						w = G.mapping(z, None)	# to save the dlatent in .npy format
						img = gen_utils.z_to_img(G, z, label, truncation_psi, noise_mode)[0]
						PIL.Image.fromarray(img, 'RGB').save(os.path.join(run_dir, f'img{idx:04d}.jpg'))
						np.save(os.path.join(run_dir, f'img{idx:04d}.npy'), w.unsqueeze(0).cpu().numpy())
		else:
				if seeds is None:
						ctx.fail('--seeds option is required when not using --projected-w')

				# Generate images.
				images = []
				for seed_idx, seed in enumerate(seeds):
						print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
						z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
						img = gen_utils.z_to_img(G, z, label, truncation_psi, noise_mode)[0]
						if save_grid:
								images.append(img)
						PIL.Image.fromarray(img, 'RGB').save(os.path.join(run_dir, f'seed{seed:04d}.jpg'))

				if save_grid:
						print('Saving image grid...')
						# We let the function infer the shape of the grid
						if (grid_width, grid_height) == (None, None):
								PIL.Image.fromarray(gen_utils.create_image_grid(np.array(images)),
																		'RGB').save(os.path.join(run_dir, 'grid.jpg'))
						# The user tells the specific shape of the grid, but one value may be None
						else:
								PIL.Image.fromarray(gen_utils.create_image_grid(np.array(images), (grid_width, grid_height)),
																		'RGB').save(os.path.join(run_dir, 'grid.jpg'))

		# Save the configuration used
		ctx.obj = {
				'network_pkl': network_pkl,
				'training_snapshot': training_snapshot,
				'snapshot_size': snapshot_size,
				'seeds': seeds,
				'truncation_psi': truncation_psi,
				'class_idx': class_idx,
				'noise_mode': noise_mode,
				'save_grid': save_grid,
				'grid_width': grid_width,
				'grid_height': grid_height,
				'run_dir': run_dir,
				'description': description,
				'projected_w': projected_w
		}
		gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


@main.command(name='random-video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--class', 'class_idx', type=gen_utils.num_range, help='List of class labels to interpolate during animation (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Video options
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Video grid width / number of columns', default=None, show_default=True)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Video grid height / number of rows', default=None, show_default=True)
@click.option('--slowdown', type=gen_utils.parse_slowdown, help='Slow down the video by this amount; will be approximated to the nearest power of 2', default='1', show_default=True)
@click.option('--duration-sec', '-sec', type=float, help='Duration length of the video', default=30.0, show_default=True)
@click.option('--fps', type=click.IntRange(min=1), help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'video'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def random_interpolation_video(
				ctx: click.Context,
				network_pkl: Union[str, os.PathLike],
				seeds: List[int],
				truncation_psi: float,
				new_center: Tuple[str, Union[int, np.ndarray]],
				class_idx: Optional[Union[int, List[int]]],
				noise_mode: str,
				anchor_latent_space: bool,
				grid_width: int,
				grid_height: int,
				slowdown: int,
				duration_sec: float,
				fps: int,
				outdir: Union[str, os.PathLike],
				description: str,
				compress: bool,
				smoothing_sec: Optional[float] = 3.0	# for Gaussian blur; won't be a command-line parameter, change at own risk
):
		"""
		Generate a random interpolation video using a pretrained network.

		Examples:

		\b
		# Generate a 30-second long, untruncated MetFaces video at 30 FPS (3 rows and 2 columns; horizontal):
		python generate.py random-video --seeds=0-5 \\
				--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

		\b
		# Generate a 60-second long, truncated 1x2 MetFaces video at 60 FPS (2 rows and 1 column; vertical):
		python generate.py random-video --trunc=0.7 --seeds=10,20 --grid-width=1 --grid-height=2 \\
				--fps=60 -sec=60 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

		"""
		print(f'Loading networks from "{network_pkl}"...')
		device = torch.device('cuda')
		with dnnlib.util.open_url(network_pkl) as f:
				G = legacy.load_network_pkl(f)['G_ema'].to(device)	# type: ignore

		if anchor_latent_space:
				gen_utils.anchor_latent_space(G)

		# Create the run dir with the given name description; add slowdown if different than the default (1)
		description = 'random-video' if len(description) == 0 else description
		description = f'{description}-{slowdown}xslowdown' if slowdown != 1 else description
		run_dir = gen_utils.make_run_dir(outdir, description)

		# Number of frames in the video and its total duration in seconds
		num_frames = int(np.rint(duration_sec * fps))
		total_duration = duration_sec * slowdown

		# generate an array of class label vectors with shape and dims in shape, each of size c_dim filled with zeros with a random one
		def labels_from_seed(seed, c_dim, shape):
				return np.eye(c_dim)[np.random.RandomState(seed).randint(np.ones([np.prod(shape)])*c_dim)].reshape([*shape, c_dim])

		print('Generating latent vectors...')
		# TODO: let another helper function handle each case, we will use it for the grid
		# If there's more than one seed provided and the shape isn't specified by the user
		if (grid_width is None and grid_height is None) and len(seeds) >= 1:
				# TODO: this can be done by another function
				# Number of images in the grid video according to the seeds provided
				num_seeds = len(seeds)
				# Get the grid width and height according to num, giving priority to the number of columns
				grid_width = max(int(np.ceil(np.sqrt(num_seeds))), 1)
				grid_height = max((num_seeds - 1) // grid_width + 1, 1)
				grid_size = (grid_width, grid_height)
				shape = [num_frames, G.z_dim]  # This is per seed
				# Get the z latents
				all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)
				# Get the class labels
				all_labels = labels_from_seeds(class_idx, G.c_dim, [num_frames, num_seeds])

		# If only one seed is provided, but the user specifies the grid shape:
		elif None not in (grid_width, grid_height) and len(seeds) == 1:
				grid_size = (grid_width, grid_height)
				shape = [num_frames, np.prod(grid_size), G.z_dim]
				# Since we have one seed, we use it to generate all latents
				all_latents = np.random.RandomState(*seeds).randn(*shape).astype(np.float32)
				# Get the class labels
				all_labels = labels_from_seeds(class_idx, G.c_dim, shape[0:2])

		# If one or more seeds are provided, and the user also specifies the grid shape:
		elif None not in (grid_width, grid_height) and len(seeds) >= 1:
				# Case is similar to the first one
				num_seeds = len(seeds)
				grid_size = (grid_width, grid_height)
				available_slots = np.prod(grid_size)
				if available_slots < num_seeds:
						diff = num_seeds - available_slots
						click.secho(f'More seeds were provided ({num_seeds}) than available spaces in the grid ({available_slots})',
												fg='red')
						click.secho(f'Removing the last {diff} seeds: {seeds[-diff:]}', fg='blue')
						seeds = seeds[:available_slots]
				shape = [num_frames, G.z_dim]
				all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)
				# Get the class labels
				all_labels = labels_from_seeds(class_idx, G.c_dim, [num_frames, num_seeds])

		else:
				ctx.fail('Error: wrong combination of arguments! Please provide either a list of seeds, one seed and the grid '
									'width and height, or more than one seed and the grid width and height')

		# Let's smooth out the random latents so that now they form a loop (and are correctly generated in a 512-dim space)
		all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[smoothing_sec * fps, 0, 0], mode='wrap')
		all_latents /= np.sqrt(np.mean(np.square(all_latents)))

		# Name of the video
		mp4_name = f'{grid_width}x{grid_height}-slerp-{slowdown}xslowdown'

		# Labels.
		'''
		def classes_to_labels(c_dim, class_idxes, num_steps, loop=True):
				idxes = np.resize(class_idxes, [len(class_idxes)+1])
				idxes[len(idxes)-1] = idxes[0]
				result = np.zeros([(len(idxes)-1) * num_steps, c_dim], dtype=np.float32)
				for i in range(0, len(idxes)-1):
						j = i + 1
						a = np.where(np.indices([num_steps, c_dim])[1]==idxes[i], 1, 0).astype(np.float32)
						af = 1-np.indices([num_steps, c_dim])[0]/num_steps
						bf = 1-af
						b = np.where(np.indices([num_steps, c_dim])[1]==idxes[j], 1, 0).astype(np.float32)
						result[i*num_steps:(i+1)*num_steps] = a*af + b*bf
				return result

		#label = torch.zeros([1, G.c_dim], device=device)
		labels = None
		if G.c_dim != 0:
				if class_idx is None:
						ctx.fail('Must specify class label with --class when using a conditional network')
				if (num_seeds // len(class_idx)) * len(class_idx) != num_seeds:
						ctx.fail('Number of latents (--seed) must be an integer multiple of number of classes specified using --class')
				labels = classes_to_labels(G.c_dim, class_idx, int(len(all_latents)//len(class_idx)*slowdown), loop=True)
				print(f"all_latents.shape = {all_latents.shape}, labels.shape = {labels.shape}, reshaping to {[*(all_latents.shape[0:-1]), G.c_dim]}")
				labels = np.repeat(labels, all_latents.shape[1], axis = 1)
				labels = labels.reshape([*(all_latents.shape[0:-1]), G.c_dim])
				print(f"after resize: labels.shape = {labels.shape}")
		else:
				if class_idx is not None:
						print('warn: --class=lbl ignored when running on an unconditional network')
		'''

		# Let's slowdown the video, if so desired
		while slowdown > 1:
				all_latents, dummy_duration_sec, dummy_num_frames = gen_utils.double_slowdown(latents=all_latents,
																																					duration=duration_sec,
																																					frames=num_frames)
				all_labels, duration_sec, num_frames = gen_utils.double_slowdown(latents=all_labels,
																																					duration=duration_sec,
																																					frames=num_frames)
				slowdown //= 2

		if new_center is None:
				def make_frame(t):
						frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
						latents = torch.from_numpy(all_latents[frame_idx]).to(device)
						if labels is None:
								label = None
						else:
								label = torch.from_numpy(all_labels[frame_idx]).to(device)
						print(label.shape)
						print(latents.shape)
						# Get the images with the labels
						images = gen_utils.z_to_img(G, latents, label, truncation_psi, noise_mode)
						# Generate the grid for this timestamp
						grid = gen_utils.create_image_grid(images, grid_size)
						# Grayscale => RGB
						if grid.shape[2] == 1:
								grid = grid.repeat(3, 2)
						return grid

		else:
				new_center, new_center_value = new_center
				# We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
				if isinstance(new_center_value, int):
						new_w_avg = gen_utils.get_w_from_seed(G, device, new_center_value, truncation_psi=1.0)	# We want the pure dlatent
				elif isinstance(new_center_value, np.ndarray):
						new_w_avg = torch.from_numpy(new_center_value).to(device)
				else:
						ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

				def make_frame(t):
						frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
						latents = torch.from_numpy(np.expand_dims(all_latents[frame_idx], 0)).to(device)
						# Do the truncation trick with this new center
						w = G.mapping(latents, None)
						w = new_w_avg + (w - new_w_avg) * truncation_psi
						# Get the images with the new center
						images = gen_utils.w_to_img(G, w, noise_mode)
						# Generate the grid for this timestamp
						grid = gen_utils.create_image_grid(images, grid_size)
						# Grayscale => RGB
						if grid.shape[2] == 1:
								grid = grid.repeat(3, 2)
						return grid

		# Generate video using the respective make_frame function
		videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
		videoclip.set_duration(total_duration)

		# Change the video parameters (codec, bitrate) if you so desire
		final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
		videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

		# Compress the video (lower file size, same resolution)
		if compress:
				gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)

		# Save the configuration used
		new_center = 'w_avg' if new_center is None else new_center
		ctx.obj = {
				'network_pkl': network_pkl,
				'seeds': seeds,
				'truncation_psi': truncation_psi,
				'new_center': new_center,
				'class_idx': class_idx,
				'noise_mode': noise_mode,
				'grid_width': grid_width,
				'grid_height': grid_height,
				'slowdown': slowdown,
				'duration_sec': duration_sec,
				'video_fps': fps,
				'run_dir': run_dir,
				'description': description,
				'compress': compress,
				'smoothing_sec': smoothing_sec
		}
		gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
		main()	# pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------

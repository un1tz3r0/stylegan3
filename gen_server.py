import asyncio
import aiohttp
from aiohttp import web

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


def init_model(network_pkl):
	device = torch.device('cuda')
	with dnnlib.util.open_url(network_pkl) as f:
		G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
	return device, G

async def render_image(device, G, 
		seeds,
		class_idxes,
		truncation_psi = 1.0,
		noise_mode = 'const',
		grid_rows = 1,
		grid_cols = 1,
		interpz = 0.0,
		interpl = 0.0
		):
	try:
		
		if len(seeds) == 1:
			seed0 = seeds[0]
			seed1 = seeds[0]
		else:
			seed0 = seeds[0]
			seed1 = seeds[1]

		if class_idxes != None::
			if G.c_dim == 0:
				raise RuntimeError("Error, cannot specify class for unconditional network")
			class_idxes = [int(word) for word in request.query['class'].split(" ")]
			if len(class_idxes) == 1:
				class_idx0 = class_idxes[0]
				class_idx1 = class_idxes[0]
			else:
				class_idx0 = class_idxes[0]
				class_idx1 = class_idxes[1]
			label = torch.zeros([grid_rows*grid_cols, G.c_dim], device=device)
			for col in range(0,grid_cols):
				a = (col/max(grid_cols-1, 1)) + interpl
				b = 1.0 - a
				for row in range(0,grid_rows):
					if class_idx0 != class_idx1:
						label[row*grid_cols+col, class_idx0] = b
						label[row*grid_cols+col, class_idx1] = a
					else:
						label[row*grid_cols+col, class_idx0] = 1
		else:
			if G.c_dim != 0:
				raise ValueError("Error, must specify class for conditional network")
			label = None
			class_idx = None

		# generate latent z interpolation
		z0 = np.random.RandomState(seed0).randn(G.z_dim)
		z1 = np.random.RandomState(seed1).randn(G.z_dim)
		zs = np.zeros([grid_rows * grid_cols, G.z_dim])
		for row in range(0, grid_cols):
			a = row/max(1,(grid_rows-1)) + interpz
			zi = gen_utils.slerp(a, z0, z1)
			for col in range(0, grid_rows):
				zs[row * grid_cols + col, :] = zi
		z = torch.from_numpy(zs).to(device)

		imgs = gen_utils.z_to_img(G, z, label, truncation_psi, noise_mode)
		img = gen_utils.create_image_grid(imgs, (grid_rows, grid_cols))
		im = PIL.Image.fromarray(img, 'RGB')
		from io import BytesIO
		with BytesIO() as stream:
			im.save(stream, "JPEG")
			return stream.value()
	except Exception as err:
		raise



async def index_handler(request):
	model_is_loaded = 'inference.G' in request.app.keys()
	return web.Response(text=f"Status: model_is_loaded={model_is_loaded}")

async def load_model_handler(request):
	''' load a network pkl into GPU memory, preparing it for inference with the /image handler.
	loaded model is available to other handlers as app['inference.G'] '''
	device = request.app['device']
	network_pkl = request.query['network_pkl']
	async def load_task():
		try:
			device = torch.device('cuda')
			with dnnlib.util.open_url(network_pkl) as f:
				G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
			request.app['inference.G'] = G
		except Exception as err:
			return "Error: {}".format(str(err))
		return "Success"
	t = asyncio.create_task(load_task())
	result = await t
	return web.Response(text=f"Result: {result}")

def clamp(a, l=0, h=1):
	if a < l: return l
	if a > h: return h
	return a

def interp_latents(a, z0, z1):
	from torch_utils.gen_utils import slerp
	a = clamp(float(a))
	return slerp(a, z0, z1)

def interp_labels(a, l0, l1):
	from torch_utils.gen_utils import lerp
	a = clamp(float(a))
	return slerp(a, z0, z1)

async def image_handler(request):
	try:
		device = request.app['device']
		if not ('inference.G' in request.app.keys()):
			return web.Response(text="Error, no network loaded, try /load?network_pkl=/path/to/network.pkl first.")
		G = request.app['inference.G']

		#seed = int(request.query['seed'])
		seeds = [int(word) for word in request.query['seed'].split(" ")]
		if len(seeds) == 1:
			seed0 = seeds[0]
			seed1 = seeds[0]
		else:
			seed0 = seeds[0]
			seed1 = seeds[1]

		if 'truncation_psi' in request.query.keys():
			truncation_psi = float(request.query['truncation_psi'])
		else:
			truncation_psi = 1.0

		if 'noise_mode' in request.query.keys():
			noise_mode = request.query['noise_mode']
			if noise_mode not in ['const', 'random', 'none']:
				return web.Response(text="Error, noise_mode, if given, must be one of 'const', 'random' or 'none', default is 'const'.")
		else:
			noise_mode = 'const'

		if 'rows' in request.query.keys():
			grid_rows = int(request.query['rows'])
		else:
			grid_rows = 1

		if 'cols' in request.query.keys():
			grid_cols = int(request.query['cols'])
		else:
			grid_cols = 1

		if 'interpz' in request.query.keys():
			interpz = float(request.query['interpz'])
		else:
			interpz = 0.0

		if 'interpl' in request.query.keys():
			interpl = float(request.query['interpl'])
		else:
			interpl = 0.0

		if 'class' in request.query.keys():
			if G.c_dim == 0:
				return web.Response(text="Error, cannot specify class for unconditional network")
			class_idxes = [int(word) for word in request.query['class'].split(" ")]
			if len(class_idxes) == 1:
				class_idx0 = class_idxes[0]
				class_idx1 = class_idxes[0]
			else:
				class_idx0 = class_idxes[0]
				class_idx1 = class_idxes[1]
			label = torch.zeros([grid_rows*grid_cols, G.c_dim], device=device)
			for col in range(0,grid_cols):
				a = (col/max(grid_cols-1, 1)) + interpl
				b = 1.0 - a
				for row in range(0,grid_rows):
					if class_idx0 != class_idx1:
						label[row*grid_cols+col, class_idx0] = b
						label[row*grid_cols+col, class_idx1] = a
					else:
						label[row*grid_cols+col, class_idx0] = 1
		else:
			if G.c_dim != 0:
				return web.Response(text="Error, must specify class for conditional network")
			label = None
			class_idx = None

		# generate latent z interpolation
		z0 = np.random.RandomState(seed0).randn(G.z_dim)
		z1 = np.random.RandomState(seed1).randn(G.z_dim)
		zs = np.zeros([grid_rows * grid_cols, G.z_dim])
		for row in range(0, grid_cols):
			a = row/max(1,(grid_rows-1)) + interpz
			zi = gen_utils.slerp(a, z0, z1)
			for col in range(0, grid_rows):
				zs[row * grid_cols + col, :] = zi
		z = torch.from_numpy(zs).to(device)

		imgs = gen_utils.z_to_img(G, z, label, truncation_psi, noise_mode)
		img = gen_utils.create_image_grid(imgs, (grid_rows, grid_cols))
		im = PIL.Image.fromarray(img, 'RGB')
		from io import BytesIO
		stream = BytesIO()
		im.save(stream, "JPEG")
		return web.Response(body=stream.getvalue(), content_type='image/jpeg')
	except Exception as err:
		return web.Response(text=f"Error: {err}")

async def init_app(*args):
		app = web.Application()
		print(f"args: {args}")
		device = torch.device('cuda')
		app['device'] = device
		app.router.add_get("/", index_handler)
		app.router.add_get("/load", load_model_handler)
		app.router.add_get("/image", image_handler)
		return app

if __name__ == "__main__":
	web.run_app(init_app(), path='/content/server.sock')

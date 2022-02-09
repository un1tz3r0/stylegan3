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

async def image_handler(request):
	device = request.app['device']
	if not 'inference.G' in request.app.keys():
		return web.Response(text="Error, no network loaded, try /load?network_pkl=/path/to/network.pkl first.")
	G = request.app['inference.G']
	seed = int(request.query['seed'])
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
	noise_mode
	if 'class' in request.query.keys():
		if G.c_dim == 0:
			return web.Response(text="Error, cannot specify class for unconditional network")
		class_idx = int(request.query['class'])
		label = torch.zeros([1, G.c_dim], device=device)
		label[:, class_idx] = 1
	else:
		if G.c_dim != 0:
			return web.Response(text="Error, must specify class for conditional network")
		label = None
		class_idx = None
	z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
	img = gen_utils.z_to_img(G, z, label, truncation_psi, noise_mode)[0]
	im = PIL.Image.fromarray(img, 'RGB')
	stream = BytesIO()
	im.save(stream, "JPEG")
	return web.Response(body=stream.getvalue(), content_type='image/jpeg')


async def init_app():
		app = web.Application()
		device = torch.device('cuda')
		app['device'] = device
		app.router.add_get("/", index_handler)
		app.router.add_get("/load", load_model_handler)
		app.router.add_get("/image", image_handler)
		return app

if __name__ == "__main__":
	web.run_app(init_app())

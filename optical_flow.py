import cv2
import numpy
import numpy as np
import torch
import imageio
# import matplotlib.pyplot as plt


class Config:
  def __init__(
      self,
      flow=None,
      pyr_scale=0.5,
      levels=3,
      winsize=15,
      iterations=3,
      poly_n=5,
      poly_sigma=1.2,
      flags=0,
      ):
    self.pyr_scale = pyr_scale
    self.levels = levels
    self.winsize = winsize
    self.iterations = iterations
    self.poly_n = poly_n
    self.poly_sigma = poly_sigma
    self.flags = flags
    self.flow = flow

def read_gray2rgb(file1, file2):
  with open(file1, "rb") as f:
    im1 = imageio.imread(f)
  with open(file2, "rb") as f:
    im2 = imageio.imread(f)

  im1rgb = cv2.cvtColor(im1,cv2.COLOR_GRAY2RGB)
  im2rgb = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)

  return im1rgb, im2rgb

def read_gray(file1, file2):
  with open(file1, "rb") as f:
    im1 = imageio.imread(f)
  with open(file2, "rb") as f:
    im2 = imageio.imread(f)
  return im1, im2

def opticalFlowFewPoints(points, im1, im2, config):
  flow = cv2.calcOpticalFlowFarneback(
    prev=im1,
    next=im2,
    flow=config.flow,
    pyr_scale=config.pyr_scale,
    levels=config.levels,
    winsize=config.winsize,
    iterations=config.iterations,
    poly_n=config.poly_n,
    poly_sigma=config.poly_sigma,
    flags=config.flags,
  )

  out_points = []
  for x, y in points:
    out = flow[x, y]
    out_x = x + out[0]
    out_y = y + out[1]
    out_points.append((int(out_x), int(out_y)))

  return out_points

def xy_to_x_y(points):
  # in: [(x,y),(x,y),..]
  # out: [x,x,..],[y,y,..]

  x = []
  y = []

  for xy in points:
    x.append(xy[0])
    y.append(xy[1])

  return x, y


if __name__ == '__main__':
  config = Config(None, 0.5, 3, 3, 1, 7, 1.5, flags=10)
  points = [(551, 369), (588, 456)]

  firstim = '24.png'
  secondim = '26.png'
  im1rgb, im2rgb = read_gray2rgb(firstim, secondim)
  im1, im2 = read_gray(firstim, secondim)
  points = opticalFlowFewPoints(points, im1, im2, config)

  print(points)

  x, y = xy_to_x_y(points)

  print(x, y)
from model import EDSR
import scipy.misc
import argparse
import data
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/General-100")
parser.add_argument("--imgsize", default=100, type=int)
parser.add_argument("--scale", default=2, type=int)
parser.add_argument("--layers", default=32, type=int)
parser.add_argument("--featuresize", default=256, type=int)
parser.add_argument("--batchsize", default=10, type=int)
parser.add_argument("--savedir", default="saved_models")
parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--numimgs", default=5, type=int)
parser.add_argument("--outdir", default="out")
parser.add_argument("--image")
args = parser.parse_args()
# 判断是否有文件夹out
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

down_size = args.imgsize // args.scale
network = EDSR(down_size, args.layers, args.featuresize, scale=args.scale)
# args.savedir = saved_models
network.resume(args.savedir)

# 获取样本库的所有图片
img_list = os.listdir(args.dataset)
for img in img_list:
    filename = os.path.basename(img)
    print(filename)
    name = filename.split('.', 1)[0]
    out_name = 'out_' + name + '.bmp'
    print(out_name)
    if filename:
        x = scipy.misc.imread(args.dataset+'/'+filename)
        inputs = x
        outputs = network.predict(x)
        # scipy.misc.imsave(args.outdir + "/input_" + args.image, inputs)
        # scipy.misc.imsave(args.outdir + "/test5.bmp", inputs)
        # scipy.misc.imsave(args.outdir + "/output_" + args.image, outputs)
        scipy.misc.imsave(args.outdir + '/' + out_name, outputs)
    else:
        print("No image argument given")

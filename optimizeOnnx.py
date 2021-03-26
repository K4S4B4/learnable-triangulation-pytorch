import onnx
import onnx.optimizer

src_onnx = 'baseline_pose2d_withConfidence2_fixedBatchSize1.onnx'
opt_onnx = 'baseline_pose2d_withConfidence2_fixedBatchSize1.opt.onnx'

# load model
model = onnx.load(src_onnx)

# optimize
model = onnx.optimizer.optimize(model, ['fuse_bn_into_conv'] )

# save optimized model
with open(opt_onnx, "wb") as f:
    f.write(model.SerializeToString())
/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class OperatorTest : public OperatorBase {
 public:
  void Init() override { x = 1; }
  void InferShape(const std::shared_ptr<Scope>& scope) const override {}
  void Run(const std::shared_ptr<Scope>& scope,
           const platform::DeviceContext& dev_ctx) const override {
    float scale = GetAttr<float>("scale");
    ASSERT_NEAR(scale, 3.14, 1e-5);
    ASSERT_EQ(scope->GetVariable(inputs_[0]), nullptr);
    ASSERT_EQ(x, 1);
    ASSERT_NE(scope->GetVariable(outputs_[0]), nullptr);
  }

 public:
  float x = 0;
};

class OpKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OpKernelTestProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is test op");
  }
};

class OpWithKernelTest : public OperatorWithKernel {
 protected:
  void InferShape(const std::vector<const Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs) const override {}
};

class CPUKernelTest : public OpKernel {
 public:
  void Compute(const KernelContext& context) const {
    float scale = context.op_.GetAttr<float>("scale");
    ASSERT_NEAR(scale, 3.14, 1e-5);
    std::cout << "this is cpu kernel" << std::endl;
    std::cout << context.op_.DebugString() << std::endl;
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP(op_with_kernel, paddle::framework::OpWithKernelTest,
            paddle::framework::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(op_with_kernel, paddle::framework::CPUKernelTest);

TEST(OpKernel, all) {
  using namespace paddle::framework;

  OpDesc op_desc;
  op_desc.set_type("op_with_kernel");
  *op_desc.mutable_inputs()->Add() = "IN1";
  *op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUDeviceContext cpu_device_context;
  auto scope = std::make_shared<Scope>();

  OperatorBase* op = paddle::framework::OpRegistry::CreateOp(op_desc);
  op->Run(scope, cpu_device_context);

  delete op;
}
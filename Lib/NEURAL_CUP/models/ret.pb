
/
inputxPlaceholder*
dtype0*
shape: 
4
PlaceholderPlaceholder*
dtype0*
shape: 
K
truncated_normal/shapeConst*
dtype0*
valueB"   @   
B
truncated_normal/meanConst*
dtype0*
valueB
 *    
D
truncated_normal/stddevConst*
dtype0*
valueB
 *wֈ>
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
Y
weightsVariable*
dtype0*
shape
:@*
	container *
shared_name 
�
weights/AssignAssignweightstruncated_normal*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0
F
weights/readIdentityweights*
_class
loc:@weights*
T0
6
zerosConst*
dtype0*
valueB@*    
T
biasesVariable*
dtype0*
shape:@*
	container *
shared_name 
s
biases/AssignAssignbiaseszeros*
validate_shape(*
_class
loc:@biases*
use_locking(*
T0
C
biases/readIdentitybiases*
_class
loc:@biases*
T0
U
MatMulMatMulinputxweights/read*
transpose_b( *
transpose_a( *
T0
(
addAddMatMulbiases/read*
T0

ReluReluadd*
T0
M
truncated_normal_1/shapeConst*
dtype0*
valueB"@   �   
D
truncated_normal_1/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_1/stddevConst*
dtype0*
valueB
 *   >
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
\
	weights_1Variable*
dtype0*
shape:	@�*
	container *
shared_name 
�
weights_1/AssignAssign	weights_1truncated_normal_1*
validate_shape(*
_class
loc:@weights_1*
use_locking(*
T0
L
weights_1/readIdentity	weights_1*
_class
loc:@weights_1*
T0
9
zeros_1Const*
dtype0*
valueB�*    
W
biases_1Variable*
dtype0*
shape:�*
	container *
shared_name 
{
biases_1/AssignAssignbiases_1zeros_1*
validate_shape(*
_class
loc:@biases_1*
use_locking(*
T0
I
biases_1/readIdentitybiases_1*
_class
loc:@biases_1*
T0
W
MatMul_1MatMulReluweights_1/read*
transpose_b( *
transpose_a( *
T0
.
add_1AddMatMul_1biases_1/read*
T0

Relu_1Reluadd_1*
T0
M
truncated_normal_2/shapeConst*
dtype0*
valueB"�   �   
D
truncated_normal_2/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_2/stddevConst*
dtype0*
valueB
 *��=
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
]
	weights_2Variable*
dtype0*
shape:
��*
	container *
shared_name 
�
weights_2/AssignAssign	weights_2truncated_normal_2*
validate_shape(*
_class
loc:@weights_2*
use_locking(*
T0
L
weights_2/readIdentity	weights_2*
_class
loc:@weights_2*
T0
9
zeros_2Const*
dtype0*
valueB�*    
W
biases_2Variable*
dtype0*
shape:�*
	container *
shared_name 
{
biases_2/AssignAssignbiases_2zeros_2*
validate_shape(*
_class
loc:@biases_2*
use_locking(*
T0
I
biases_2/readIdentitybiases_2*
_class
loc:@biases_2*
T0
Y
MatMul_2MatMulRelu_1weights_2/read*
transpose_b( *
transpose_a( *
T0
.
add_2AddMatMul_2biases_2/read*
T0

Relu_2Reluadd_2*
T0
M
truncated_normal_3/shapeConst*
dtype0*
valueB"�   @   
D
truncated_normal_3/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_3/stddevConst*
dtype0*
valueB
 *��=
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
\
	weights_3Variable*
dtype0*
shape:	�@*
	container *
shared_name 
�
weights_3/AssignAssign	weights_3truncated_normal_3*
validate_shape(*
_class
loc:@weights_3*
use_locking(*
T0
L
weights_3/readIdentity	weights_3*
_class
loc:@weights_3*
T0
8
zeros_3Const*
dtype0*
valueB@*    
V
biases_3Variable*
dtype0*
shape:@*
	container *
shared_name 
{
biases_3/AssignAssignbiases_3zeros_3*
validate_shape(*
_class
loc:@biases_3*
use_locking(*
T0
I
biases_3/readIdentitybiases_3*
_class
loc:@biases_3*
T0
Y
MatMul_3MatMulRelu_2weights_3/read*
transpose_b( *
transpose_a( *
T0
.
add_3AddMatMul_3biases_3/read*
T0

Relu_3Reluadd_3*
T0
M
truncated_normal_4/shapeConst*
dtype0*
valueB"@      
D
truncated_normal_4/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_4/stddevConst*
dtype0*
valueB
 *   >
~
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0
S
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0
[
	weights_4Variable*
dtype0*
shape
:@*
	container *
shared_name 
�
weights_4/AssignAssign	weights_4truncated_normal_4*
validate_shape(*
_class
loc:@weights_4*
use_locking(*
T0
L
weights_4/readIdentity	weights_4*
_class
loc:@weights_4*
T0
8
zeros_4Const*
dtype0*
valueB*    
V
biases_4Variable*
dtype0*
shape:*
	container *
shared_name 
{
biases_4/AssignAssignbiases_4zeros_4*
validate_shape(*
_class
loc:@biases_4*
use_locking(*
T0
I
biases_4/readIdentitybiases_4*
_class
loc:@biases_4*
T0
Y
MatMul_4MatMulRelu_3weights_4/read*
transpose_b( *
transpose_a( *
T0
.
add_4AddMatMul_4biases_4/read*
T0
4
ToInt64CastPlaceholder*

DstT0	*

SrcT0
)
xentropy/ShapeShapeToInt64*
T0	
`
xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogitsadd_4ToInt64*
T0*
Tlabels0	
3
ConstConst*
dtype0*
valueB: 
I
xentropy_meanMeanxentropy/xentropyConst*
T0*
	keep_dims( 
C
global_step/initial_valueConst*
dtype0*
value	B : 
U
global_stepVariable*
dtype0*
shape: *
	container *
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/initial_value*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0
R
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0
0
gradients/ShapeShapexentropy_mean*
T0
<
gradients/ConstConst*
dtype0*
valueB
 *  �?
A
gradients/FillFillgradients/Shapegradients/Const*
T0
X
*gradients/xentropy_mean_grad/Reshape/shapeConst*
dtype0*
valueB:
t
$gradients/xentropy_mean_grad/ReshapeReshapegradients/Fill*gradients/xentropy_mean_grad/Reshape/shape*
T0
G
"gradients/xentropy_mean_grad/ShapeShapexentropy/xentropy*
T0
|
!gradients/xentropy_mean_grad/TileTile$gradients/xentropy_mean_grad/Reshape"gradients/xentropy_mean_grad/Shape*
T0
I
$gradients/xentropy_mean_grad/Shape_1Shapexentropy/xentropy*
T0
E
$gradients/xentropy_mean_grad/Shape_2Shapexentropy_mean*
T0
P
"gradients/xentropy_mean_grad/ConstConst*
dtype0*
valueB: 
�
!gradients/xentropy_mean_grad/ProdProd$gradients/xentropy_mean_grad/Shape_1"gradients/xentropy_mean_grad/Const*
T0*
	keep_dims( 
R
$gradients/xentropy_mean_grad/Const_1Const*
dtype0*
valueB: 
�
#gradients/xentropy_mean_grad/Prod_1Prod$gradients/xentropy_mean_grad/Shape_2$gradients/xentropy_mean_grad/Const_1*
T0*
	keep_dims( 
P
&gradients/xentropy_mean_grad/Maximum/yConst*
dtype0*
value	B :
�
$gradients/xentropy_mean_grad/MaximumMaximum#gradients/xentropy_mean_grad/Prod_1&gradients/xentropy_mean_grad/Maximum/y*
T0
~
%gradients/xentropy_mean_grad/floordivDiv!gradients/xentropy_mean_grad/Prod$gradients/xentropy_mean_grad/Maximum*
T0
h
!gradients/xentropy_mean_grad/CastCast%gradients/xentropy_mean_grad/floordiv*

DstT0*

SrcT0
z
$gradients/xentropy_mean_grad/truedivDiv!gradients/xentropy_mean_grad/Tile!gradients/xentropy_mean_grad/Cast*
T0
?
gradients/zeros_like	ZerosLikexentropy/xentropy:1*
T0
b
/gradients/xentropy/xentropy_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������
�
+gradients/xentropy/xentropy_grad/ExpandDims
ExpandDims$gradients/xentropy_mean_grad/truediv/gradients/xentropy/xentropy_grad/ExpandDims/dim*
T0
v
$gradients/xentropy/xentropy_grad/mulMul+gradients/xentropy/xentropy_grad/ExpandDimsxentropy/xentropy:1*
T0
6
gradients/add_4_grad/ShapeShapeMatMul_4*
T0
=
gradients/add_4_grad/Shape_1Shapebiases_4/read*
T0
}
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1
�
gradients/add_4_grad/SumSum$gradients/xentropy/xentropy_grad/mul*gradients/add_4_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0
�
gradients/add_4_grad/Sum_1Sum$gradients/xentropy/xentropy_grad/mul,gradients/add_4_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
T0
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
T0
�
gradients/MatMul_4_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyweights_4/read*
transpose_b(*
transpose_a( *
T0
�
 gradients/MatMul_4_grad/MatMul_1MatMulRelu_3-gradients/add_4_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
�
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*
T0
�
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
T0
m
gradients/Relu_3_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_3*
T0
6
gradients/add_3_grad/ShapeShapeMatMul_3*
T0
=
gradients/add_3_grad/Shape_1Shapebiases_3/read*
T0
}
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1
�
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0
�
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0
�
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyweights_3/read*
transpose_b(*
transpose_a( *
T0
�
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/add_3_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*
T0
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
T0
m
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*
T0
6
gradients/add_2_grad/ShapeShapeMatMul_2*
T0
=
gradients/add_2_grad/Shape_1Shapebiases_2/read*
T0
}
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyweights_2/read*
transpose_b(*
transpose_a( *
T0
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/add_2_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
T0
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
T0
m
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0
6
gradients/add_1_grad/ShapeShapeMatMul_1*
T0
=
gradients/add_1_grad/Shape_1Shapebiases_1/read*
T0
}
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyweights_1/read*
transpose_b(*
transpose_a( *
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
i
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0
2
gradients/add_grad/ShapeShapeMatMul*
T0
9
gradients/add_grad/Shape_1Shapebiases/read*
T0
w
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1

gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
`
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
f
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyweights/read*
transpose_b(*
transpose_a( *
T0
�
gradients/MatMul_grad/MatMul_1MatMulinputx+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
J
GradientDescent/learning_rateConst*
dtype0*
valueB
 *��8
�
3GradientDescent/update_weights/ApplyGradientDescentApplyGradientDescentweightsGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@weights*
use_locking( *
T0
�
2GradientDescent/update_biases/ApplyGradientDescentApplyGradientDescentbiasesGradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_class
loc:@biases*
use_locking( *
T0
�
5GradientDescent/update_weights_1/ApplyGradientDescentApplyGradientDescent	weights_1GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@weights_1*
use_locking( *
T0
�
4GradientDescent/update_biases_1/ApplyGradientDescentApplyGradientDescentbiases_1GradientDescent/learning_rate/gradients/add_1_grad/tuple/control_dependency_1*
_class
loc:@biases_1*
use_locking( *
T0
�
5GradientDescent/update_weights_2/ApplyGradientDescentApplyGradientDescent	weights_2GradientDescent/learning_rate2gradients/MatMul_2_grad/tuple/control_dependency_1*
_class
loc:@weights_2*
use_locking( *
T0
�
4GradientDescent/update_biases_2/ApplyGradientDescentApplyGradientDescentbiases_2GradientDescent/learning_rate/gradients/add_2_grad/tuple/control_dependency_1*
_class
loc:@biases_2*
use_locking( *
T0
�
5GradientDescent/update_weights_3/ApplyGradientDescentApplyGradientDescent	weights_3GradientDescent/learning_rate2gradients/MatMul_3_grad/tuple/control_dependency_1*
_class
loc:@weights_3*
use_locking( *
T0
�
4GradientDescent/update_biases_3/ApplyGradientDescentApplyGradientDescentbiases_3GradientDescent/learning_rate/gradients/add_3_grad/tuple/control_dependency_1*
_class
loc:@biases_3*
use_locking( *
T0
�
5GradientDescent/update_weights_4/ApplyGradientDescentApplyGradientDescent	weights_4GradientDescent/learning_rate2gradients/MatMul_4_grad/tuple/control_dependency_1*
_class
loc:@weights_4*
use_locking( *
T0
�
4GradientDescent/update_biases_4/ApplyGradientDescentApplyGradientDescentbiases_4GradientDescent/learning_rate/gradients/add_4_grad/tuple/control_dependency_1*
_class
loc:@biases_4*
use_locking( *
T0
�
GradientDescent/updateNoOp4^GradientDescent/update_weights/ApplyGradientDescent3^GradientDescent/update_biases/ApplyGradientDescent6^GradientDescent/update_weights_1/ApplyGradientDescent5^GradientDescent/update_biases_1/ApplyGradientDescent6^GradientDescent/update_weights_2/ApplyGradientDescent5^GradientDescent/update_biases_2/ApplyGradientDescent6^GradientDescent/update_weights_3/ApplyGradientDescent5^GradientDescent/update_biases_3/ApplyGradientDescent6^GradientDescent/update_weights_4/ApplyGradientDescent5^GradientDescent/update_biases_4/ApplyGradientDescent
x
GradientDescent/valueConst^GradientDescent/update*
dtype0*
_class
loc:@global_step*
value	B :
|
GradientDescent	AssignAddglobal_stepGradientDescent/value*
_class
loc:@global_step*
use_locking( *
T0
6
InTopKInTopKadd_4Placeholder*
k*
T0
,
CastCastInTopK*

DstT0*

SrcT0

5
Const_1Const*
dtype0*
valueB: 
3
SumSumCastConst_1*
T0*
	keep_dims( 
�
initNoOp^weights/Assign^biases/Assign^weights_1/Assign^biases_1/Assign^weights_2/Assign^biases_2/Assign^weights_3/Assign^biases_3/Assign^weights_4/Assign^biases_4/Assign^global_step/Assign
8

save/ConstConst*
dtype0*
valueB Bmodel
�
save/save/tensor_namesConst*
dtype0*�
value|BzBbiasesBbiases_1Bbiases_2Bbiases_3Bbiases_4Bglobal_stepBweightsB	weights_1B	weights_2B	weights_3B	weights_4
\
save/save/shapes_and_slicesConst*
dtype0*)
value BB B B B B B B B B B B 
�
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesbiasesbiases_1biases_2biases_3biases_4global_stepweights	weights_1	weights_2	weights_3	weights_4*
T
2
c
save/control_dependencyIdentity
save/Const
^save/save*
_class
loc:@save/Const*
T0
M
save/restore_slice/tensor_nameConst*
dtype0*
valueB Bbiases
K
"save/restore_slice/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_sliceRestoreSlice
save/Constsave/restore_slice/tensor_name"save/restore_slice/shape_and_slice*
preferred_shard���������*
dt0
~
save/AssignAssignbiasessave/restore_slice*
validate_shape(*
_class
loc:@biases*
use_locking(*
T0
Q
 save/restore_slice_1/tensor_nameConst*
dtype0*
valueB Bbiases_1
M
$save/restore_slice_1/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_1RestoreSlice
save/Const save/restore_slice_1/tensor_name$save/restore_slice_1/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_1Assignbiases_1save/restore_slice_1*
validate_shape(*
_class
loc:@biases_1*
use_locking(*
T0
Q
 save/restore_slice_2/tensor_nameConst*
dtype0*
valueB Bbiases_2
M
$save/restore_slice_2/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_2RestoreSlice
save/Const save/restore_slice_2/tensor_name$save/restore_slice_2/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_2Assignbiases_2save/restore_slice_2*
validate_shape(*
_class
loc:@biases_2*
use_locking(*
T0
Q
 save/restore_slice_3/tensor_nameConst*
dtype0*
valueB Bbiases_3
M
$save/restore_slice_3/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_3RestoreSlice
save/Const save/restore_slice_3/tensor_name$save/restore_slice_3/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_3Assignbiases_3save/restore_slice_3*
validate_shape(*
_class
loc:@biases_3*
use_locking(*
T0
Q
 save/restore_slice_4/tensor_nameConst*
dtype0*
valueB Bbiases_4
M
$save/restore_slice_4/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_4RestoreSlice
save/Const save/restore_slice_4/tensor_name$save/restore_slice_4/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_4Assignbiases_4save/restore_slice_4*
validate_shape(*
_class
loc:@biases_4*
use_locking(*
T0
T
 save/restore_slice_5/tensor_nameConst*
dtype0*
valueB Bglobal_step
M
$save/restore_slice_5/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_5RestoreSlice
save/Const save/restore_slice_5/tensor_name$save/restore_slice_5/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_5Assignglobal_stepsave/restore_slice_5*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0
P
 save/restore_slice_6/tensor_nameConst*
dtype0*
valueB Bweights
M
$save/restore_slice_6/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_6RestoreSlice
save/Const save/restore_slice_6/tensor_name$save/restore_slice_6/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_6Assignweightssave/restore_slice_6*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0
R
 save/restore_slice_7/tensor_nameConst*
dtype0*
valueB B	weights_1
M
$save/restore_slice_7/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_7RestoreSlice
save/Const save/restore_slice_7/tensor_name$save/restore_slice_7/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_7Assign	weights_1save/restore_slice_7*
validate_shape(*
_class
loc:@weights_1*
use_locking(*
T0
R
 save/restore_slice_8/tensor_nameConst*
dtype0*
valueB B	weights_2
M
$save/restore_slice_8/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_8RestoreSlice
save/Const save/restore_slice_8/tensor_name$save/restore_slice_8/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_8Assign	weights_2save/restore_slice_8*
validate_shape(*
_class
loc:@weights_2*
use_locking(*
T0
R
 save/restore_slice_9/tensor_nameConst*
dtype0*
valueB B	weights_3
M
$save/restore_slice_9/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_9RestoreSlice
save/Const save/restore_slice_9/tensor_name$save/restore_slice_9/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_9Assign	weights_3save/restore_slice_9*
validate_shape(*
_class
loc:@weights_3*
use_locking(*
T0
S
!save/restore_slice_10/tensor_nameConst*
dtype0*
valueB B	weights_4
N
%save/restore_slice_10/shape_and_sliceConst*
dtype0*
valueB B 
�
save/restore_slice_10RestoreSlice
save/Const!save/restore_slice_10/tensor_name%save/restore_slice_10/shape_and_slice*
preferred_shard���������*
dt0
�
save/Assign_10Assign	weights_4save/restore_slice_10*
validate_shape(*
_class
loc:@weights_4*
use_locking(*
T0
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10"

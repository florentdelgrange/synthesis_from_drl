Ď

��
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
�
RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0	
�
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
�
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	�*
dtype0
�
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#QNetwork/EncodingNetwork/dense/bias
�
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:�*
dtype0
�
'QNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'QNetwork/EncodingNetwork/dense_1/kernel
�
;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
%QNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%QNetwork/EncodingNetwork/dense_1/bias
�
9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_1/bias*
_output_shapes	
:�*
dtype0
�
'QNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'QNetwork/EncodingNetwork/dense_2/kernel
�
;QNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
%QNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%QNetwork/EncodingNetwork/dense_2/bias
�
9QNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_2/bias*
_output_shapes	
:�*
dtype0
�
QNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameQNetwork/dense_3/kernel
�
+QNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
QNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameQNetwork/dense_3/bias
{
)QNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
?

train_step
metadata
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

ref
1

ref
1

	_wrapped_policy
 


saved_policy
y

train_step
metadata
model_variables
_all_assets

signatures
#_self_saveable_object_factories
tr
VARIABLE_VALUE
Variable_1T_all_assets/0/ref/_wrapped_policy/saved_policy/train_step/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

0
 
 
��
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_1/kernel[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_1/bias[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_2/kernel[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_2/bias[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEQNetwork/dense_3/kernel[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEQNetwork/dense_3/bias[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/7/.ATTRIBUTES/VARIABLE_VALUE

1
5

_q_network
#_self_saveable_object_factories
�
_encoder
_q_value_layer
	variables
 trainable_variables
!regularization_losses
"	keras_api
##_self_saveable_object_factories
 
�
$_postprocessing_layers
%	variables
&trainable_variables
'regularization_losses
(	keras_api
#)_self_saveable_object_factories
�

kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
#._self_saveable_object_factories
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
 trainable_variables
!regularization_losses
#4_self_saveable_object_factories
 

50
61
72
83
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
%	variables
&trainable_variables
'regularization_losses
#>_self_saveable_object_factories
 

0
1

0
1
 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
*	variables
+trainable_variables
,regularization_losses
#D_self_saveable_object_factories
 
 

0
1
 
 
 
 
w
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
#I_self_saveable_object_factories
�

kernel
bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
#N_self_saveable_object_factories
�

kernel
bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
#S_self_saveable_object_factories
�

kernel
bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
#X_self_saveable_object_factories
 

50
61
72
83
 
 
 
 
 
 
 
 
 
 
 
 
 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
#^_self_saveable_object_factories
 

0
1

0
1
 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
#d_self_saveable_object_factories
 

0
1

0
1
 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
#j_self_saveable_object_factories
 

0
1

0
1
 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
#p_self_saveable_object_factories
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0_discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0_observationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
action_0_rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0_step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/bias'QNetwork/EncodingNetwork/dense_1/kernel%QNetwork/EncodingNetwork/dense_1/bias'QNetwork/EncodingNetwork/dense_2/kernel%QNetwork/EncodingNetwork/dense_2/biasQNetwork/dense_3/kernelQNetwork/dense_3/bias*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_20135507
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_20135519
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_20135541
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_20135534
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOp;QNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOp+QNetwork/dense_3/kernel/Read/ReadVariableOp)QNetwork/dense_3/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_20136178
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable
Variable_1%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/bias'QNetwork/EncodingNetwork/dense_1/kernel%QNetwork/EncodingNetwork/dense_1/bias'QNetwork/EncodingNetwork/dense_2/kernel%QNetwork/EncodingNetwork/dense_2/biasQNetwork/dense_3/kernelQNetwork/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_20136218��
�
8
&__inference_signature_wrapper_20135519

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_20135514*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�>
�	
0__inference_polymorphic_distribution_fn_19766448
	step_type

reward
discount
observationP
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�M
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	�B
/qnetwork_dense_3_matmul_readvariableop_resource:	�>
0qnetwork_dense_3_biasadd_readvariableop_resource:
identity	

identity_1	

identity_2	

identity_3��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp�'QNetwork/dense_3/BiasAdd/ReadVariableOp�&QNetwork/dense_3/MatMul/ReadVariableOpw
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0	*
_output_shapes
: m

Identity_1Identity Categorical/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: P

Identity_3IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
f
&__inference_signature_wrapper_20135534
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_20135526^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
8
&__inference_get_initial_state_20135513

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�`
�	
*__inference_polymorphic_action_fn_19766691
	step_type

reward
discount
observationP
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�M
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	�B
/qnetwork_dense_3_matmul_readvariableop_resource:	�>
0qnetwork_dense_3_biasadd_readvariableop_resource:
identity	

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp�'QNetwork/dense_3/BiasAdd/ReadVariableOp�&QNetwork/dense_3/MatMul/ReadVariableOpw
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB j
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/subSub%Deterministic/sample/Reshape:output:0 Categorical/mode/ArgMax:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/AbsAbs1Deterministic/log_prob/Deterministic/prob/sub:z:0*
T0	*#
_output_shapes
:����������
3Deterministic/log_prob/Deterministic/prob/LessEqual	LessEqual1Deterministic/log_prob/Deterministic/prob/Abs:y:0Deterministic/atol:output:0*
T0	*#
_output_shapes
:����������
.Deterministic/log_prob/Deterministic/prob/CastCast7Deterministic/log_prob/Deterministic/prob/LessEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:����������
Deterministic/log_prob/LogLog2Deterministic/log_prob/Deterministic/prob/Cast:y:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:����������
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 \
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:���������k

Identity_1IdentityDeterministic/log_prob/Log:y:0^NoOp*
T0*#
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�n
�
*__inference_polymorphic_action_fn_20135458
	time_step
time_step_1
time_step_2
time_step_3
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:

identity_1	

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_restored_function_body_20135340T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R {
+Deterministic/mode/Deterministic/mean/ShapeShape StatefulPartitionedCall:output:0*
T0	*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:�
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastTo StatefulPartitionedCall:output:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0	*#
_output_shapes
:���������V
Deterministic_1/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R V
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:���������y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������>
ShapeShape	time_step*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskR
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:T
random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R T
random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0	*#
_output_shapes
:���������*

seed*T
shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_1ConcatV2strided_slice:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *����Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    �
random_uniform_1/RandomUniformRandomUniformconcat_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: �
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:����������
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:����������
IdentityIdentityrandom_uniform:output:0
^time_step^time_step_1^time_step_2^time_step_3*
T0	*#
_output_shapes
:���������[
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*#
_output_shapes
:���������@
Shape_1Shape	time_step*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    T
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=x
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������x
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0	*#
_output_shapes
:���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: G
Shape_2ShapeGreaterEqual:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :[
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
: O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_2:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeGreaterEqual:z:0concat_2:output:0*
T0
*#
_output_shapes
:����������
SelectV2SelectV2Reshape:output:0 StatefulPartitionedCall:output:1random_uniform_1:z:0*
T0*#
_output_shapes
:���������[
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*#
_output_shapes
:���������`

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_2IdentitySelectV2:output:0^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�
(
&__inference_signature_wrapper_19766617�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_19766615*(
_construction_contextkEagerRuntime*
_input_shapes 
�n
�
*__inference_polymorphic_action_fn_20135765
	step_type

reward
discount
observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:

identity_1	

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_restored_function_body_20135340T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R {
+Deterministic/mode/Deterministic/mean/ShapeShape StatefulPartitionedCall:output:0*
T0	*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:�
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastTo StatefulPartitionedCall:output:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0	*#
_output_shapes
:���������V
Deterministic_1/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R V
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:���������y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������>
ShapeShape	step_type*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskR
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:T
random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R T
random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0	*#
_output_shapes
:���������*

seed*T
shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_1ConcatV2strided_slice:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *����Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    �
random_uniform_1/RandomUniformRandomUniformconcat_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: �
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:����������
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:����������
IdentityIdentityrandom_uniform:output:0	^discount^observation^reward
^step_type*
T0	*#
_output_shapes
:���������[
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*#
_output_shapes
:���������@
Shape_1Shape	step_type*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    T
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=x
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������x
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0	*#
_output_shapes
:���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: G
Shape_2ShapeGreaterEqual:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :[
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
: O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_2:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeGreaterEqual:z:0concat_2:output:0*
T0
*#
_output_shapes
:����������
SelectV2SelectV2Reshape:output:0 StatefulPartitionedCall:output:1random_uniform_1:z:0*
T0*#
_output_shapes
:���������[
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*#
_output_shapes
:���������`

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_2IdentitySelectV2:output:0^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�q
�
0__inference_polymorphic_distribution_fn_20136066
	step_type

reward
discount
observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:

identity_1	

identity_2	

identity_3	

identity_4��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_restored_function_body_20135340T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R {
+Deterministic/mode/Deterministic/mean/ShapeShape StatefulPartitionedCall:output:0*
T0	*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:�
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastTo StatefulPartitionedCall:output:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0	*#
_output_shapes
:���������V
Deterministic_1/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R V
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:���������y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������>
ShapeShape	step_type*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskR
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:T
random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R T
random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0	*#
_output_shapes
:���������*

seed*T
shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_1ConcatV2strided_slice:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *����Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    �
random_uniform_1/RandomUniformRandomUniformconcat_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: �
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:����������
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:����������
IdentityIdentityrandom_uniform:output:0	^discount^observation^reward
^step_type*
T0	*#
_output_shapes
:���������[
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*#
_output_shapes
:���������@
Shape_1Shape	step_type*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    T
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=x
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������x
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0	*#
_output_shapes
:���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: G
Shape_2ShapeGreaterEqual:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :[
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
: O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_2:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeGreaterEqual:z:0concat_2:output:0*
T0
*#
_output_shapes
:����������
SelectV2SelectV2Reshape:output:0 StatefulPartitionedCall:output:1random_uniform_1:z:0*
T0*#
_output_shapes
:���������[
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*#
_output_shapes
:���������V
Deterministic_2/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R V
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R ]

Identity_1IdentityDeterministic_2/atol:output:0^NoOp*
T0	*
_output_shapes
: `

Identity_2Identityclip_by_value_2:z:0^NoOp*
T0	*#
_output_shapes
:���������]

Identity_3IdentityDeterministic_2/rtol:output:0^NoOp*
T0	*
_output_shapes
: ^

Identity_4IdentitySelectV2:output:0^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�0
�
$__inference__traced_restore_20136218
file_prefix#
assignvariableop_variable:	 '
assignvariableop_1_variable_1:	 K
8assignvariableop_2_qnetwork_encodingnetwork_dense_kernel:	�E
6assignvariableop_3_qnetwork_encodingnetwork_dense_bias:	�N
:assignvariableop_4_qnetwork_encodingnetwork_dense_1_kernel:
��G
8assignvariableop_5_qnetwork_encodingnetwork_dense_1_bias:	�N
:assignvariableop_6_qnetwork_encodingnetwork_dense_2_kernel:
��G
8assignvariableop_7_qnetwork_encodingnetwork_dense_2_bias:	�=
*assignvariableop_8_qnetwork_dense_3_kernel:	�6
(assignvariableop_9_qnetwork_dense_3_bias:
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEBT_all_assets/0/ref/_wrapped_policy/saved_policy/train_step/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_qnetwork_encodingnetwork_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_qnetwork_encodingnetwork_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp:assignvariableop_4_qnetwork_encodingnetwork_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_qnetwork_encodingnetwork_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp:assignvariableop_6_qnetwork_encodingnetwork_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_qnetwork_encodingnetwork_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_qnetwork_dense_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_qnetwork_dense_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
&__inference_signature_wrapper_20135507
discount
observation

reward
	step_type
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity	

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_20135479k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�
>
,__inference_function_with_signature_20135514

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_get_initial_state_20135513*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
l
,__inference_function_with_signature_19766458
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference_<lambda>_19766453`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
�
+__inference_restored_function_body_20135340
	time_step
time_step_1
time_step_2
time_step_3
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity	

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_polymorphic_action_fn_19766691k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�`
�	
*__inference_polymorphic_action_fn_19766776
time_step_step_type
time_step_reward
time_step_discount
time_step_observationP
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�M
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	�B
/qnetwork_dense_3_matmul_readvariableop_resource:	�>
0qnetwork_dense_3_biasadd_readvariableop_resource:
identity	

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp�'QNetwork/dense_3/BiasAdd/ReadVariableOp�&QNetwork/dense_3/MatMul/ReadVariableOpw
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB j
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/subSub%Deterministic/sample/Reshape:output:0 Categorical/mode/ArgMax:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/AbsAbs1Deterministic/log_prob/Deterministic/prob/sub:z:0*
T0	*#
_output_shapes
:����������
3Deterministic/log_prob/Deterministic/prob/LessEqual	LessEqual1Deterministic/log_prob/Deterministic/prob/Abs:y:0Deterministic/atol:output:0*
T0	*#
_output_shapes
:����������
.Deterministic/log_prob/Deterministic/prob/CastCast7Deterministic/log_prob/Deterministic/prob/LessEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:����������
Deterministic/log_prob/LogLog2Deterministic/log_prob/Deterministic/prob/Cast:y:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:����������
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 \
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:���������k

Identity_1IdentityDeterministic/log_prob/Log:y:0^NoOp*
T0*#
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step/observation
�
8
&__inference_get_initial_state_19766466

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
,__inference_function_with_signature_19766564
	step_type

reward
discount
observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity	

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_polymorphic_action_fn_19766546`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation
�
8
&__inference_get_initial_state_19766585

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
(
&__inference_signature_wrapper_20135541�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_20135537*(
_construction_contextkEagerRuntime*
_input_shapes 
�
.
,__inference_function_with_signature_20135537�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference_<lambda>_20134904*(
_construction_contextkEagerRuntime*
_input_shapes 
^

__inference_<lambda>_20134904*(
_construction_contextkEagerRuntime*
_input_shapes 
�
8
&__inference_signature_wrapper_19766472

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_19766469*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
.
,__inference_function_with_signature_19766615�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference_<lambda>_19766613*(
_construction_contextkEagerRuntime*
_input_shapes 
^

__inference_<lambda>_19766613*(
_construction_contextkEagerRuntime*
_input_shapes 
�
d
__inference_<lambda>_20134901!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�
d
__inference_<lambda>_19766453!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�
8
&__inference_get_initial_state_20136086

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�$
�
!__inference__traced_save_20136178
file_prefix'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	D
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_dense_2_kernel_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_2_bias_read_readvariableop6
2savev2_qnetwork_dense_3_kernel_read_readvariableop4
0savev2_qnetwork_dense_3_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEBT_all_assets/0/ref/_wrapped_policy/saved_policy/train_step/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB[_all_assets/0/ref/_wrapped_policy/saved_policy/model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopBsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableopBsavev2_qnetwork_encodingnetwork_dense_2_kernel_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_2_bias_read_readvariableop2savev2_qnetwork_dense_3_kernel_read_readvariableop0savev2_qnetwork_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*d
_input_shapesS
Q: : : :	�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%	!

_output_shapes
:	�: 


_output_shapes
::

_output_shapes
: 
�o
�
*__inference_polymorphic_action_fn_20135888
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:

identity_1	

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltime_step_step_typetime_step_rewardtime_step_discounttime_step_observationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_restored_function_body_20135340T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R {
+Deterministic/mode/Deterministic/mean/ShapeShape StatefulPartitionedCall:output:0*
T0	*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:�
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastTo StatefulPartitionedCall:output:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0	*#
_output_shapes
:���������V
Deterministic_1/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R V
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:���������y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������H
ShapeShapetime_step_step_type*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskR
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:T
random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R T
random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0	*#
_output_shapes
:���������*

seed*T
shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_1ConcatV2strided_slice:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *����Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    �
random_uniform_1/RandomUniformRandomUniformconcat_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: �
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:����������
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:����������
IdentityIdentityrandom_uniform:output:0^time_step_discount^time_step_observation^time_step_reward^time_step_step_type*
T0	*#
_output_shapes
:���������[
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*#
_output_shapes
:���������J
Shape_1Shapetime_step_step_type*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    T
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype0*

seed**
seed2�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L=x
GreaterEqualGreaterEqualepsilon_rng/Mul:z:0GreaterEqual/y:output:0*
T0*#
_output_shapes
:���������x
SelectSelectGreaterEqual:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0	*#
_output_shapes
:���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: G
Shape_2ShapeGreaterEqual:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :[
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
: O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_2:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeGreaterEqual:z:0concat_2:output:0*
T0
*#
_output_shapes
:����������
SelectV2SelectV2Reshape:output:0 StatefulPartitionedCall:output:1random_uniform_1:z:0*
T0*#
_output_shapes
:���������[
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0	*#
_output_shapes
:���������S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*#
_output_shapes
:���������`

Identity_1Identityclip_by_value_2:z:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_2IdentitySelectV2:output:0^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step/observation
�
f
&__inference_signature_wrapper_19766463
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_19766458`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
�
,__inference_function_with_signature_20135479
	step_type

reward
discount
observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity	

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_polymorphic_action_fn_20135458k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation
�
l
,__inference_function_with_signature_20135526
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference_<lambda>_20134901^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�`
�	
*__inference_polymorphic_action_fn_19766546
	time_step
time_step_1
time_step_2
time_step_3P
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�M
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�S
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:
��O
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	�B
/qnetwork_dense_3_matmul_readvariableop_resource:	�>
0qnetwork_dense_3_biasadd_readvariableop_resource:
identity	

identity_1��5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp�'QNetwork/dense_3/BiasAdd/ReadVariableOp�&QNetwork/dense_3/MatMul/ReadVariableOpw
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB j
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/subSub%Deterministic/sample/Reshape:output:0 Categorical/mode/ArgMax:output:0*
T0	*#
_output_shapes
:����������
-Deterministic/log_prob/Deterministic/prob/AbsAbs1Deterministic/log_prob/Deterministic/prob/sub:z:0*
T0	*#
_output_shapes
:����������
3Deterministic/log_prob/Deterministic/prob/LessEqual	LessEqual1Deterministic/log_prob/Deterministic/prob/Abs:y:0Deterministic/atol:output:0*
T0	*#
_output_shapes
:����������
.Deterministic/log_prob/Deterministic/prob/CastCast7Deterministic/log_prob/Deterministic/prob/LessEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:����������
Deterministic/log_prob/LogLog2Deterministic/log_prob/Deterministic/prob/Cast:y:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:����������
NoOpNoOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 \
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:���������k

Identity_1IdentityDeterministic/log_prob/Log:y:0^NoOp*
T0*#
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�
>
,__inference_function_with_signature_19766469

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_get_initial_state_19766466*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
&__inference_signature_wrapper_19766582
discount
observation

reward
	step_type
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity	

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2	*
_collective_manager_ids
 *2
_output_shapes 
:���������:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *5
f0R.
,__inference_function_with_signature_19766564`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0_discount:0���������
>
0/observation-
action_0_observation:0���������
0
0/reward$
action_0_reward:0���������
6
0/step_type'
action_0_step_type:0���������6
action,
StatefulPartitionedCall:0	���������D
info/log_probability,
StatefulPartitionedCall:1���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:��
�

train_step
metadata
_all_assets

signatures

qaction
rdistribution
sget_initial_state
tget_metadata
uget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
`

vaction
wget_initial_state
xget_train_step
yget_metadata"
signature_map
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
3
	_wrapped_policy"
_generic_user_object
"
_generic_user_object
0

saved_policy"
_generic_user_object
�

train_step
metadata
model_variables
_all_assets

signatures
#_self_saveable_object_factories

zaction
{distribution
|get_initial_state
}get_metadata
~get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
0"
trackable_list_wrapper
c

action
�get_initial_state
�get_train_step
�get_metadata"
signature_map
 "
trackable_dict_wrapper
8:6	�2%QNetwork/EncodingNetwork/dense/kernel
2:0�2#QNetwork/EncodingNetwork/dense/bias
;:9
��2'QNetwork/EncodingNetwork/dense_1/kernel
4:2�2%QNetwork/EncodingNetwork/dense_1/bias
;:9
��2'QNetwork/EncodingNetwork/dense_2/kernel
4:2�2%QNetwork/EncodingNetwork/dense_2/bias
*:(	�2QNetwork/dense_3/kernel
#:!2QNetwork/dense_3/bias
'
1"
trackable_list_wrapper
S

_q_network
#_self_saveable_object_factories"
_generic_user_object
�
_encoder
_q_value_layer
	variables
 trainable_variables
!regularization_losses
"	keras_api
##_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
�
$_postprocessing_layers
%	variables
&trainable_variables
'regularization_losses
(	keras_api
#)_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
#._self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
 trainable_variables
!regularization_losses
#4_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
<
50
61
72
83"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
%	variables
&trainable_variables
'regularization_losses
#>_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
*	variables
+trainable_variables
,regularization_losses
#D_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
#I_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

kernel
bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
#N_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

kernel
bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
#S_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

kernel
bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
#X_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
#^_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
#d_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
#j_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
#p_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�2�
*__inference_polymorphic_action_fn_20135765
*__inference_polymorphic_action_fn_20135888�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_polymorphic_distribution_fn_20136066�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_get_initial_state_20136086�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_20134904"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_20134901"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_20135507
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_20135519
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_20135534"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_20135541"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_polymorphic_action_fn_19766691
*__inference_polymorphic_action_fn_19766776�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_polymorphic_distribution_fn_19766448�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_get_initial_state_19766585�
���
FullArgSpec
args�
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_19766613"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_19766453"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_19766582
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_19766472
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_19766463"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_19766617"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 <
__inference_<lambda>_19766453�

� 
� "� 	5
__inference_<lambda>_19766613�

� 
� "� <
__inference_<lambda>_20134901�

� 
� "� 	5
__inference_<lambda>_20134904�

� 
� "� S
&__inference_get_initial_state_19766585)"�
�
�

batch_size 
� "� S
&__inference_get_initial_state_20136086)"�
�
�

batch_size 
� "� �
*__inference_polymorphic_action_fn_19766691����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep&
action�
action���������	
state� V
infoN�K

PolicyInfo=
log_probability*�'
info/log_probability����������
*__inference_polymorphic_action_fn_19766776����
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������
� 
� "���

PolicyStep&
action�
action���������	
state� V
infoN�K

PolicyInfo=
log_probability*�'
info/log_probability����������
*__inference_polymorphic_action_fn_20135765����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep&
action�
action���������	
state� V
infoN�K

PolicyInfo=
log_probability*�'
info/log_probability����������
*__inference_polymorphic_action_fn_20135888����
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������
� 
� "���

PolicyStep&
action�
action���������	
state� V
infoN�K

PolicyInfo=
log_probability*�'
info/log_probability����������
0__inference_polymorphic_distribution_fn_19766448����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep�
action������
`
B�?

atol� 	

loc����������	

rtol� 	
J�G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
�
j
parameters
� 
�
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
state� I
infoA�>

PolicyInfo0
log_probability�
info/log_probability �
0__inference_polymorphic_distribution_fn_20136066����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep�
action������
`
B�?

atol� 	

loc����������	

rtol� 	
L�I

allow_nan_statsp

namejDeterministic_2_1

validate_argsp 
�
j
parameters
� 
�
jname+tfp.distributions.Deterministic_ACTTypeSpec 
state� V
infoN�K

PolicyInfo=
log_probability*�'
info/log_probability���������Z
&__inference_signature_wrapper_197664630�

� 
� "�

int64�
int64 	a
&__inference_signature_wrapper_1976647270�-
� 
&�#
!

batch_size�

batch_size "� �
&__inference_signature_wrapper_19766582����
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"o�l
&
action�
action���������	
B
info/log_probability*�'
info/log_probability���������>
&__inference_signature_wrapper_19766617�

� 
� "� �
&__inference_signature_wrapper_20135507����
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"o�l
&
action�
action���������	
B
info/log_probability*�'
info/log_probability���������a
&__inference_signature_wrapper_2013551970�-
� 
&�#
!

batch_size�

batch_size "� Z
&__inference_signature_wrapper_201355340�

� 
� "�

int64�
int64 	>
&__inference_signature_wrapper_20135541�

� 
� "� 
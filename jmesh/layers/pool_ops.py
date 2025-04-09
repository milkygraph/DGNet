import jittor as jt
import numpy as np
import torch

using_cpu = True


def np_dtype_to_torch(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.uint8:
        return torch.uint8
    else:
        raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")


def to_numpy(x):
    import torch

    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, jt.jittor_core.Var):
        return x.numpy()
    else:
        raise TypeError(f"Unsupported input type for to_numpy: {type(x)}")


class PoolFuncV2(jt.Function):
    def execute(self, feats, mask, adj):
        if using_cpu:
            return self.execute_cpu(feats, mask, adj)

        head = r"""
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        """
        src = r"""
        __global__ void pool(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* feats,
                            const int* adj,
                            const int* mask,
                            float * out_feats,
                            int* indices){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                out_feats[index] = feats[index];
                indices[index] = f;

                if(mask[n*F+f]==1){
                    for(int i=0;i<3;i++){
                        int j = n*F*3+f*3+i;
                        int a = adj[j];
                        int k = n*F+a;
                        if(a>=0 && mask[k]==0){
                            int k_index = n*C*F+c*F+a;
                            if(out_feats[index]<feats[k_index]){
                                out_feats[index] = feats[k_index];
                                indices[index] = a;
                            }
                        }
                    }
                        
                }
            }
        }
        @alias(feats_t,in0);
        @alias(adj_t,in1);
        @alias(mask_t,in2);
        @alias(output_t,out0);
        @alias(indices_t,out1)
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        pool<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,adj_t_p,mask_t_p,output_t_p,indices_t_p);
        """
        N, C, F = feats.shape
        feats, indices = jt.code(
            [(N, C, F), (N, C, F)],
            [feats.dtype, adj.dtype],
            inputs=[feats, adj, mask],
            cuda_header=head,
            cuda_src=src,
        )

        self.indices = indices

        return feats

    def execute_cpu(self, feats, mask, adj):
        print("Using CPU pool")

        original_is_torch = isinstance(feats, torch.Tensor)

        feats = to_numpy(feats)
        mask = to_numpy(mask)
        adj = to_numpy(adj)

        N, C, F = feats.shape
        out_feats = feats.copy()
        indices = np.broadcast_to(np.arange(F).reshape(1, 1, F), (N, C, F)).astype(
            np.int32
        )

        for i in range(3):
            a_idx = adj[:, :, i]  # shape: (N, F)
            valid_center = mask == 1  # shape: (N, F)
            valid_neighbor = (a_idx >= 0) & (
                mask[np.arange(N)[:, None], a_idx] == 0
            )  # shape: (N, F)
            valid = valid_center & valid_neighbor

            if not np.any(valid):
                continue

            n_idx, f_idx = np.where(valid)
            a = a_idx[n_idx, f_idx]

            for c in range(C):
                neighbor_vals = feats[n_idx, c, a]
                current_vals = out_feats[n_idx, c, f_idx]
                update_mask = neighbor_vals > current_vals
                out_feats[n_idx[update_mask], c, f_idx[update_mask]] = neighbor_vals[
                    update_mask
                ]
                indices[n_idx[update_mask], c, f_idx[update_mask]] = a[update_mask]

        self.indices = indices

        # Convert back to torch.Tensor if original input was torch
        if original_is_torch:
            torch_dtype = np_dtype_to_torch(feats.dtype)
            out_feats = torch.tensor(out_feats, dtype=torch_dtype)
            self.indices = torch.tensor(indices, dtype=torch.int64)

        return out_feats

    def grad(self, grad_output):
        head = r"""
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        """
        src = r"""
        __global__ void pool_backward(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* grad_output,
                            const int* indices,
                            float * out_grad){
            CUDA_KERNEL_LOOP(index, nthreads){
                int c = (index / F) % C;
                int n =  index / F / C;
                int src_f = indices[index];
                atomicAdd(out_grad+n*F*C+c*F+src_f,grad_output[index]);
            }
        }
        @alias(feats_t,in0);
        @alias(indices_t,in1);
        @alias(output_t,out0);
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        cudaMemsetAsync(output_t_p,0,output_t->size);
        pool_backward<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,indices_t_p,output_t_p);
        """
        print("Using grad")
        grad_feats = jt.code(
            grad_output.shape,
            grad_output.dtype,
            inputs=[grad_output, self.indices],
            cuda_header=head,
            cuda_src=src,
        )

        return grad_feats


pool_funcv2 = PoolFuncV2.apply


class UnPoolFuncV2(jt.Function):
    def execute(self, feats, mask, adj, is_bilinear=1):
        if using_cpu:
            return self.execute_cpu(feats, mask, adj, is_bilinear)
        head = (
            r"""
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        """
            + f"""
        #define IS_BILI {is_bilinear}
        """
        )
        src = r"""
        __global__ void pool(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* feats,
                            const int* adj,
                            const int* mask,
                            float * out_feats,
                            int* indices){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                int i_index = n*F*3+f*3;
                out_feats[index] = feats[index];
                if(c==0){
                    indices[i_index] = f;
                    indices[i_index+1] = -1;
                    indices[i_index+2] = -1;
                }
                if(mask[n*F+f]==0){
                    float val = 0;
                    int count = 0;
                    for(int i=0;i<3;i++){
                        int j = n*F*3+f*3+i;
                        int a = adj[j];
                        int k = n*F+a;
                        if(a>=F){
                            printf("Error!!!!!!!!!\n");
                        }
                        if(a>=0 && mask[k]==1){
                            int k_index = n*C*F+c*F+a;
                            if (c==0)
                                indices[i_index+count] = a;
                            val +=feats[k_index];
                            count += 1;   
                            if(!IS_BILI)
                                break;
                        }
                    }
                    if(count>0)
                        out_feats[index] = val/count;
                    else
                        out_feats[index] = 0.0f;
                }
            }
        }
        @alias(feats_t,in0);
        @alias(adj_t,in1);
        @alias(mask_t,in2);
        @alias(output_t,out0);
        @alias(indices_t,out1)
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        pool<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,adj_t_p,mask_t_p,output_t_p,indices_t_p);
        """
        N, C, F = feats.shape
        feats, indices = jt.code(
            [(N, C, F), (N, F, 3)],
            [feats.dtype, adj.dtype],
            inputs=[feats, adj, mask],
            cuda_header=head,
            cuda_src=src,
        )

        self.indices = indices

        return feats

    def execute_cpu(self, feats, mask, adj, is_bilinear=0):
        print("Using CPU unpool")

        N, C, F = feats.shape
        feats = to_numpy(feats)
        mask = to_numpy(mask)
        adj = to_numpy(adj)

        out_np = feats.copy()
        indices_np = np.broadcast_to(np.arange(F).reshape(1, 1, F), (N, C, F)).astype(
            np.int32
        )

        for n in range(N):
            for f in range(F):
                if mask[n, f] != 1:
                    continue
                for i in range(3):
                    a = adj[n, f, i]
                    if a < 0 or mask[n, a] != 0:
                        continue
                    for c in range(C):
                        if feats[n, c, a] > out_np[n, c, f]:
                            out_np[n, c, f] = feats[n, c, a]
                            indices_np[n, c, f] = a

        self.indices = indices_np
        return out_np

    def grad(self, grad_output):
        head = r"""
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        """
        src = r"""
        __global__ void pool_backward(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* grad_output,
                            const int* indices,
                            float * out_grad){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                int i_index = n*F*3+f*3;
                int count = (indices[i_index]>0)+(indices[i_index+1]>0)+(indices[i_index+2]>0);
                for(int i=0;i<count;i++){
                    atomicAdd(out_grad+n*F*C+c*F+indices[i_index+i],grad_output[index]/count);
                }                
            }
        }
        @alias(feats_t,in0);
        @alias(indices_t,in1);
        @alias(output_t,out0);
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        cudaMemsetAsync(output_t_p,0,output_t->size);
        pool_backward<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,indices_t_p,output_t_p);
        """

        print("pool backward")
        grad_feats = jt.code(
            grad_output.shape,
            grad_output.dtype,
            inputs=[grad_output, self.indices],
            cuda_header=head,
            cuda_src=src,
        )

        return grad_feats


unpool_funcv2 = UnPoolFuncV2.apply

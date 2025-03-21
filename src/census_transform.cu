/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "internal.h"

#include <cuda_runtime.h>

#include "types.h"
#include "host_utility.h"

namespace sgm
{
namespace
{

static constexpr int WINDOW_WIDTH  = 9;
static constexpr int WINDOW_HEIGHT = 7;

static constexpr int BLOCK_SIZE = 128;
static constexpr int LINES_PER_BLOCK = 16;

template <typename T>
__global__ void census_transform_kernel(uint64_t* dest, const T* src, int width, int height, int pitch)
{
	using pixel_type = T;
	using feature_type = uint64_t;

	static const int SMEM_BUFFER_SIZE = WINDOW_HEIGHT + 1;

	const int half_kw = WINDOW_WIDTH / 2;
	const int half_kh = WINDOW_HEIGHT / 2;

	__shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int x0 = blockIdx.x * (BLOCK_SIZE - WINDOW_WIDTH + 1) - half_kw;
	const int y0 = blockIdx.y * LINES_PER_BLOCK;

	for (int i = 0; i < WINDOW_HEIGHT; ++i) {
		const int x = x0 + tid, y = y0 - half_kh + i;
		pixel_type value = 0;
		if (0 <= x && x < width && 0 <= y && y < height) {
			value = src[x + y * pitch];
		}
		smem_lines[i][tid] = value;
	}
	__syncthreads();

#pragma unroll
	for (int i = 0; i < LINES_PER_BLOCK; ++i) {
		if (i + 1 < LINES_PER_BLOCK) {
			// Load to smem
			const int x = x0 + tid, y = y0 + half_kh + i + 1;
			pixel_type value = 0;
			if (0 <= x && x < width && 0 <= y && y < height) {
				value = src[x + y * pitch];
			}
			const int smem_x = tid;
			const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
			smem_lines[smem_y][smem_x] = value;
		}

		if (half_kw <= tid && tid < BLOCK_SIZE - half_kw) {
			// Compute and store
			const int x = x0 + tid, y = y0 + i;
			if (half_kw <= x && x < width - half_kw && half_kh <= y && y < height - half_kh) {
				const int smem_x = tid;
				const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
				const auto a = smem_lines[smem_y][smem_x];
				feature_type f = 0;
				for (int dy = -half_kh; dy <= half_kh; ++dy) {
					for (int dx = -half_kw; dx <= half_kw; ++dx) {
						if (dx != 0 && dy != 0) {
							const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
							const int smem_x1 = smem_x + dx;
							const auto b = smem_lines[smem_y1][smem_x1];
							f = (f << 1) | (a > b);
						}
					}
				}
				dest[x + y * width] = f;
			}
		}
		__syncthreads();
	}
}

template <typename T>
__global__ void symmetric_census_kernel(uint32_t* dest, const T* src, int width, int height, int pitch)
{
	using pixel_type = T;
	using feature_type = uint32_t;

	static const int SMEM_BUFFER_SIZE = WINDOW_HEIGHT + 1;

	const int half_kw = WINDOW_WIDTH  / 2;
	const int half_kh = WINDOW_HEIGHT / 2;

	__shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int x0 = blockIdx.x * (BLOCK_SIZE - WINDOW_WIDTH + 1) - half_kw;
	const int y0 = blockIdx.y * LINES_PER_BLOCK;

	for(int i = 0; i < WINDOW_HEIGHT; ++i){
		const int x = x0 + tid, y = y0 - half_kh + i;
		pixel_type value = 0;
		if(0 <= x && x < width && 0 <= y && y < height){
			value = src[x + y * pitch];
		}
		smem_lines[i][tid] = value;
	}
	__syncthreads();

#pragma unroll
	for(int i = 0; i < LINES_PER_BLOCK; ++i){
		if(i + 1 < LINES_PER_BLOCK){
			// Load to smem
			const int x = x0 + tid, y = y0 + half_kh + i + 1;
			pixel_type value = 0;
			if(0 <= x && x < width && 0 <= y && y < height){
				value = src[x + y * pitch];
			}
			const int smem_x = tid;
			const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
			smem_lines[smem_y][smem_x] = value;
		}

		if(half_kw <= tid && tid < BLOCK_SIZE - half_kw){
			// Compute and store
			const int x = x0 + tid, y = y0 + i;
			if(half_kw <= x && x < width - half_kw && half_kh <= y && y < height - half_kh){
				const int smem_x = tid;
				const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
				feature_type f = 0;
				for(int dy = -half_kh; dy < 0; ++dy){
					const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					for(int dx = -half_kw; dx <= half_kw; ++dx){
						const int smem_x1 = smem_x + dx;
						const int smem_x2 = smem_x - dx;
						const auto a = smem_lines[smem_y1][smem_x1];
						const auto b = smem_lines[smem_y2][smem_x2];
						f = (f << 1) | (a > b);
					}
				}
				for(int dx = -half_kw; dx < 0; ++dx){
					const int smem_x1 = smem_x + dx;
					const int smem_x2 = smem_x - dx;
					const auto a = smem_lines[smem_y][smem_x1];
					const auto b = smem_lines[smem_y][smem_x2];
					f = (f << 1) | (a > b);
				}
				dest[x + y * width] = f;
			}
		}
		__syncthreads();
	}
}

} // namespace

namespace details
{

void census_transform(const DeviceImage& src, DeviceImage& dst, CensusType type)
{
	const int w = src.cols;
	const int h = src.rows;

	const int w_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
	const int h_per_block = LINES_PER_BLOCK;
	const dim3 gdim(divUp(w, w_per_block), divUp(h, h_per_block));
	const dim3 bdim(BLOCK_SIZE);

	dst.create(h, w, type == CensusType::CENSUS_9x7 ? SGM_64U : SGM_32U);

	if (type == CensusType::CENSUS_9x7) {
		if (src.type == SGM_8U)
			census_transform_kernel<<<gdim, bdim>>>(dst.ptr<uint64_t>(), src.ptr<uint8_t>(), w, h, src.step);
		else if (src.type == SGM_16U)
			census_transform_kernel<<<gdim, bdim>>>(dst.ptr<uint64_t>(), src.ptr<uint16_t>(), w, h, src.step);
		else
			census_transform_kernel<<<gdim, bdim>>>(dst.ptr<uint64_t>(), src.ptr<uint32_t>(), w, h, src.step);
	}
	else if (type == CensusType::SYMMETRIC_CENSUS_9x7) {
		if (src.type == SGM_8U)
			symmetric_census_kernel<<<gdim, bdim>>>(dst.ptr<uint32_t>(), src.ptr<uint8_t>(), w, h, src.step);
		else if (src.type == SGM_16U)
			symmetric_census_kernel<<<gdim, bdim>>>(dst.ptr<uint32_t>(), src.ptr<uint16_t>(), w, h, src.step);
		else
			symmetric_census_kernel<<<gdim, bdim>>>(dst.ptr<uint32_t>(), src.ptr<uint32_t>(), w, h, src.step);
	}

	CUDA_CHECK(cudaGetLastError());
}

} // namespace details
} // namespace sgm

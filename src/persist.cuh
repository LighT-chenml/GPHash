#pragma once

__device__ void mfence()
{
    __threadfence_system();
}

void fence()
{
    asm volatile("mfence": : :"memory");
}
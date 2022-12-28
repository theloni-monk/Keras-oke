FROM flml/flashlight:cpu-latest

ENV BACKEND CPU
ENV MKLROOT /opt/intel/mkl
ENV ArrayFire_DIR /opt/arrayfire/share/ArrayFire/cmake
ENV DNNL_DIR /opt/dnnl/dnnl_lnx_2.0.0_cpu_iomp/lib/cmake/dnnl

WORKDIR ~/kerasoke
ADD ./* ./

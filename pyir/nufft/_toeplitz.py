# static void decompose_dims(unsigned int N, long dims2[2 * N], long ostrs2[2 * N], long istrs2[2 * N],
#         const long factors[N], const long odims[N + 1], const long ostrs[N + 1], const long idims[N], const long istrs[N])
# {
#     long prod = 1;

#     for (unsigned int i = 0; i < N; i++) {

#         long f2 = idims[i] / factors[i];

#         assert(0 == idims[i] % factors[i]);
#         assert(odims[i] == idims[i] / factors[i]);

#         dims2[1 * N + i] = factors[i];
#         dims2[0 * N + i] = f2;

#         istrs2[0 * N + i] = istrs[i] * factors[i];
#         istrs2[1 * N + i] = istrs[i];

#         ostrs2[0 * N + i] = ostrs[i];
#         ostrs2[1 * N + i] = ostrs[N] * prod;

#         prod *= factors[i];
#     }

#     assert(odims[N] == prod);
# }

# void md_decompose2(unsigned int N, const long factors[N],
#         const long odims[N + 1], const long ostrs[N + 1], void* out,
#         const long idims[N], const long istrs[N], const void* in, size_t size)
# {
#     long dims2[2 * N];
#     long ostrs2[2 * N];
#     long istrs2[2 * N];

#     decompose_dims(N, dims2, ostrs2, istrs2, factors, odims, ostrs, idims, istrs);

#     md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
# }

# void md_decompose(unsigned int N, const long factors[N], const long odims[N + 1],
#         void* out, const long idims[N], const void* in, size_t size)
# {
#     long ostrs[N + 1];
#     md_calc_strides(N + 1, ostrs, odims, size);

#     long istrs[N];
#     md_calc_strides(N, istrs, idims, size);

#     md_decompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
# }


def MD_BIT(x):
    """equivalent to the following C code:
        define MD_BIT(x) (1ul << (x))
    """
    return 1 << x


def MD_IS_SET(x, y):
    """equivalent to the following C code:
        define MD_IS_SET(x, y)  ((x) & MD_BIT(y))
    """
    return x & MD_BIT(y)



struct sample_data {

    unsigned int N;
    const long* strs;
    complex float* out;
    void* data;
    sample_fun_t fun;
};

static void sample_kernel(void* _data, const long pos[])
{
    struct sample_data* data = _data;
    data->out[md_calc_offset(data->N, data->strs, pos)] = data->fun(data->data, pos);
}



static void md_loop_r(unsigned int D, const long dim[D], long pos[D], void* data, md_loop_fun_t fun)
{
    if (0 == D) {

        fun(data, pos);
        return;
    }

    D--;

    for (pos[D] = 0; pos[D] < dim[D]; pos[D]++)
        md_loop_r(D, dim, pos, data, fun);
}
#   for z in range(nz):
#       for y in range(ny):
#           for x in range(nx):
#               fun(data, )

/**
 * Generic function which loops over all dimensions and calls a given
 * function passing the current indices as argument.
 *
 * Runs fun( data, position ) for all position in dim
 *
 */
void md_loop(unsigned int D, const long dim[D], void* data, md_loop_fun_t fun)
{
    long pos[D];
    md_loop_r(D, dim, pos, data, fun);
}


void md_zsample(unsigned int N, const long dims[N], complex float* out, void* data, sample_fun_t fun)
{
    struct sample_data sdata;

    sdata.N = N;

    long strs[N];
    md_calc_strides(N, strs, dims, 1);  // we use size = 1 here
    sdata.strs = strs;

    sdata.out = out;
    sdata.data = data;
    sdata.fun = fun;

    md_loop(N, dims, &sdata, sample_kernel);
}

struct gradient_data {

    unsigned int N;
    const complex float* grad;
};

static complex float gradient_kernel(void* _data, const long pos[])
{
    struct gradient_data* data = _data;

    complex float val = 0.;

    for (unsigned int i = 0; i < data->N; i++)
        val += pos[i] * data->grad[i];

    return val;
}


void md_zgradient(unsigned int N, const long dims[N], complex float* out, const complex float grad[N])
{
    struct gradient_data data = { N, grad };
    md_zsample(N, dims, out, &data, gradient_kernel);
}

void centered_gradient(unsigned int N, const long dims[N], const complex float grad[N], complex float* out)
{
    md_zgradient(N, dims, out, grad);

    long dims0[N];
    md_singleton_dims(N, dims0);

    long strs0[N];
    md_calc_strides(N, strs0, dims0, CFL_SIZE);

    complex float cn = 0.;

    for (unsigned int n = 0; n < N; n++)
         cn -= grad[n] * (float)dims[n] / 2.;

    long strs[N];
    md_calc_strides(N, strs, dims, CFL_SIZE);

    md_zadd2(N, dims, strs, out, strs, out, strs0, &cn);
}


def linear_phase(unsigned int N, const long dims[N], const float pos[N], complex float* out):
    grad = np.zeros(N, dtype=np.complex64)
    for n in range(N):
        grad[n] = 2j * np.pi * pos[n] / dims[n]
    # TODO: finish
    # centered_gradient(N, dims, grad, out);
    # md_zmap(N, dims, out, out, cexpf);
    return out


def compute_linphases(N, img_dims):
    img_dims = np.asarray(img_dims)
    ND = N + 3
    if img_dims.ndim != ND:
        raise ValueError("invalid img_dims")
    shifts = np.zeros((8, 3))
    s = 0
    for i in range(8):
        skip = false
        for j in range(3):
            shifts[s][j] = 0.
            if MD_IS_SET(i, j):
                skip = skip or (1 == img_dims[j])
                shifts[s][j] = -0.5
        if not skip:
            s += 1
    lph_dims = img_dims.copy()
    lph_dims[N] = s
    linphase = np.zeros(lph_dims, dtype=np.complex64)

    for i in range(s):
        linear_phase(ND, img_dims, shifts[i, :],
                     linphase + i * md_calc_size(ND, img_dims))
    return linphase



def compute_psf2(img_dims, traj, weights):
    traj2 = 2 * traj
    img_dims2 = 2 * np.asarray(img_dims2)
    psft = compute_psf(ND, img2_dims, trj_dims, traj2, weights)
    psft = fftuc(psft)
    ndim = len(img_dims2)
    scale = 4**ndim
    psft *= scale
    factors = 2*np.ones(ndim)
    # psf = md_decompose(N + 0, factors, psf_dims, img2_dims, psft, CFL_SIZE);


def compute_psf(
    G = nufft_create(N, ksp_dims1, img2_dims, trj_dims, traj, NULL,
                     nufft_conf_defaults, false);
    ones = np.ones(ksp_dims1, dtype=np.complex64)
    ones = weights * ones
    ones = np.conj(weights) * ones
    psft = G.H * ones
    return psft
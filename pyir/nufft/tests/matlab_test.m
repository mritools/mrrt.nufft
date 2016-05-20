ig = image_geom('nx', 32, 'ny', 28, 'dx', 1, 'offsets', 'dsp');
ig.mask = ig.circ(1+ig.nx/2, 1+ig.ny/2) > 0;
N = ig.dim;
J = [6 6];

[kspace omega wi] = mri_trajectory('spiral0', {}, N, ig.fov, {'voronoi'});
warn 'todo: crudely fixing dcf at edge of spiral'
wi(350:end) = wi(350);

im pl 3 3
if im
    clf, im subplot 1, plot(omega(:,1), omega(:,2), '.')
    axis([-1 1 -1 1]*pi), axis square
    titlef('%d k-space samples', size(omega,1))
end
nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};

cpu etic
Gn = Gnufft(ig.mask, {omega, nufft_args{:}});

if 0
   nufft_args_sparse = {N, J, 2*N, N/2, 'minmax:kb'}
   Gn_sparse = Gnufft(ig.mask, {omega, nufft_args_sparse{:}});
   
   %newfft_args_sparse = {'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','minmax:kb'}
   %Gnewfft_sparse = Gnewfft(ig.mask, {omega, newfft_args_sparse{:}})

   data = nufft_init(omega,N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb')
   save OLD_Gtest_minmax_table0 data 
   data = nufft_init(omega,N, J, 2*N, N/2, 'minmax:kb')
   data.p = full(data.p.arg.G)
   save OLD_Gtest_minmax_sparse data 
   st_minmax_sparse = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','minmax:kb','phasing','complex')
   st_minmax_sparse_real = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','minmax:kb','phasing','real')
   st_minmax_table0 = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','table0','ktype','minmax:kb','phasing','complex')
   st_minmax_table0_real = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','table0','ktype','minmax:kb','phasing','real')
   
   st_beatty_sparse = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','kb:beatty','phasing','complex')
   st_beatty_sparse_real = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','kb:beatty','phasing','real')
   st_beatty_table0 = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','table0','ktype','kb:beatty','phasing','complex')
   st_beatty_table0_real = newfft(omega, N, 'Jd', J, 'Kd', 2*N, 'n_shift', N/2, 'mode','sparse','ktype','kb:beatty','phasing','real')

    data = st_minmax_sparse.data
    data.p = full(data.p.arg.G)
    save Gtest_minmax_sparse data kspace wi
    data = st_minmax_sparse_real.data
    data.p = full(data.p.arg.G)
    save Gtest_minmax_sparse_real data kspace wi
    data = st_minmax_table0.data
    save Gtest_minmax_table0 data kspace wi
    data = st_minmax_table0_real.data
    save Gtest_minmax_table0_real data kspace wi

    data = st_beatty_table0.data
    save Gtest_beatty_table0 data kspace wi
    data = st_beatty_table0_real.data
    save Gtest_beatty_table0_real data kspace wi
    data = st_beatty_sparse.data
    data.p = full(data.p.arg.G)
    save Gtest_beatty_sparse data kspace wi
    data = st_beatty_sparse_real.data
    data.p = full(data.p.arg.G)
    save Gtest_beatty_sparse_real data kspace wi


end

ktype = Gn.arg.st.ktype
om = Gn.arg.st.om
phase_shift = Gn.arg.st.phase_shift
n_shift = Gn.arg.st.n_shift
alpha1 = Gn.arg.st.alpha{1}
alpha2 = Gn.arg.st.alpha{2}
beta1 = Gn.arg.st.beta{1}
beta2 = Gn.arg.st.beta{2}
Nd = Gn.arg.st.Nd
Jd = Gn.arg.st.Jd
Kd = Gn.arg.st.Kd
Ld = Gn.arg.st.Ld
M = Gn.arg.st.M
sn = Gn.arg.st.sn
h1 = Gn.arg.st.h{1}
h2 = Gn.arg.st.h{2}
p = full(Gn_sparse.st.p)
save Gnufft_testkb ktype om phase_shift n_shift alpha1 alpha2 beta1 beta2 Nd Jd Kd Ld M h1 h2 sn p kspace wi

function newfft_data_to_mat(st,filename,kspace,wi)
    om = st.data.om;
    ktype = st.data.ktype;
    phasing = st.data.phasing;
    phase_before = st.data.phase_before;
    phase_after = st.data.phase_after;
    n_shift = st.data.n_shift;
    dd = st.data.dd;
    Nd = st.data.Nd;
    Jd = st.data.Jd;
    Kd = st.data.Kd;
    Ld = st.data.oversample;
    M = size(om,1)
    Nmid = st.data.Nmid
    sn = st.data.sn;
    is_kaiser_scale = st.data.is_kaiser_scale
    h1 = st.data.h{1};
    h2 = st.data.h{2};
    if strfind('minmax',ktype)
        alpha1 = st.data.alpha{1};
        beta1 = st.data.beta{1};
        if dd > 1
            alpha2 = st.data.alpha{2};
            beta2 = st.data.beta{2};
        end
        if dd > 2
            alpha3 = st.data.alpha{3};
            beta3 = st.data.beta{3};
        end
    end
    if strfind('kaiser',ktype)
        kb_m1 = st.data.kb_m{1};
        kb_alf1 = st.data.kb_alf{1};
        if dd > 1
            kb_m2 = st.data.kb_m{2};
            kb_alf2 = st.data.kb_alf{2};
        end
        if dd > 2
            kb_m3 = st.data.kb_m{3};
            kb_alf3 = st.data.kb_alf{3};
        end
    end    
    mode = st.data.mode;
    if strfind('sparse',mode)
        p = full(st.data.p);
    end


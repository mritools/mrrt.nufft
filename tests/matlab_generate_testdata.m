modes = {'sparse', 'table0', 'table1'};

phasings = {'complex', 'real'};
% 'flipreal' no longer needed thanks to floor(N/2)
% 'none' is for internal table only

Nd_list = [20 10 8];
for id=1 %0:3 %2
	if id == 0
		Nd = 1 + Nd_list(1); % test odd case
	else
		Nd = Nd_list(1:id);
	end
	dd = length(Nd);

	rng(0)
	om = 3 * 2 * pi * (rand(100,length(Nd)) - 0.5);
%	om = sort(om);
%	om = linspace(-1,1,601)' * 3*pi;
%	om = [-8:8]'/2 * pi;
	%om = 'epi';
	x0 = rand([Nd 1]);
%	x0 = zeros([Nd 1]); x0(1) = 1; % unit vector for testing
%	x0 = ones([Nd 1]);

	% exact
	st_e = newfft(om, Nd, 'mode', 'exact');
	Xe = st_e.fft(x0);
	xe = st_e.adj(Xe);
	if 0
		pe = st_e.p;
		equivs(pe * x0(:), Xe)
		equivs(reshape(pe' * Xe, [Nd 1]), xe)
    end

    if 0
        ktypes = {{'linear', 'Jd', 2*ones(1,dd)}, ...
            'minmax:unif', ...
            {'minmax:user', 'alpha', num2cell(ones(1,dd)), ...
                'beta', num2cell(0.5 * ones(1,dd))}, ...
            {'diric', 'Jd', 2*Nd-0, 'oversample', []}, ...
            'minmax:kb', 'minmax:tuned', ...
            'kb:minmax', 'kb:beatty', ...
            {'kb:user', 'kb_m', 0*Nd, 'kb_alf', 2.34 * 6 + 0*Nd}
            };
    else
        ktypes = {'minmax:kb', 'minmax:tuned', ...
            'kb:minmax', 'kb:beatty', ...
            {'kb:user', 'kb_m', 0*Nd, 'kb_alf', 2.34 * 6 + 0*Nd}
            };
        
    end
    
    for ii=4 %1:length(ktypes) % skip poor ones  %4
            ktype = ktypes{ii};
            if ~iscell(ktype), ktype = {ktype};  end
    
	for jj=2 %1:length(modes)  %1
	for ip=2 %1:length(phasings)  %2
        mode = modes{jj}
        phasing = phasings{ip}
        
        st = newfft(st_e.om, st_e.Nd, 'phasing', phasings{ip}, ...
			'mode', modes{jj}, 'ktype', ktype{:});

		if streq(st.phasing, 'complex') && streq(st.mode, 'table1')
			continue
		end

%		pr minmax(st.sn)
		pad = @(s,n) [s blanks(n-length(s))];
		key = [sprintf('%2d ', st.Jd(1)) st.ktype];
		key = [st.mode ' ' num2str(id) st.phasing(1) ' ' pad(key,16)];
		Xs = st.fft(x0);
		err_for = max_percent_diff(Xe, Xs, key)
%		plot(abs(Xe), abs(Xs), 'o'), prompt

		xs = st.adj(Xe);
		err_adj = max_percent_diff(xe, xs, key)
        
        if ~streq(ktype{1},'diric')
            %diric case too large.  don't save that one to disk
            if streq(mode,'sparse')
                st.p = full(st.p.arg.G);
            end
            data = st.data;
            eval(sprintf('save newwfft_test_%dD_%s_%s_%s.mat data x0 xe Xe xs Xs err_for err_adj',dd,strrep(ktype{1},':','_'), mode, phasing'))
        end
        
	end % ip
	end % jj
	end % ii
end
% input tdm, vocabulary and class information
% tdm, vocabfull clvar (filenames are handy for text identification)
kill
% output two tables Class Semantics with most likely terms within each
% class & Misclass semantics with most likely terms within each
% misclassified document including id in dtm and target and output (misclassified) class
% cd('C:\Users\KLN\Documents\projects\humclass')
% classification datasets
cd('/home/kln/projects/humclass')
fpath = '/home/kln/projects/humclass/figures';
%load('ntTest.mat')
load('chineseTest.mat')
%load('chinese2Test.mat')

% prune tdm
[tdm,vocabfull,dtmstat] = prune_mat(tdm,vocabfull,.95);
disp(dtmstat)

disp('Press key to continue');
input('');
% plot covariance matrix
h1 = plot_cov(tdm);
%saveFig(strcat(fpath,'/covariance_tdm.png'))

run('presskey.m')

% # for presentation of NB model
%subplot(1,2,1),
%run('plot_dtm.m')
%subplot(1,2,2),
%run('plot_prior.m')
%saveFig('/home/kln/projects/humclass/figures/dtm_priors.png')
%run('presskey.m')


% ### class semantics
cl = unique(clvar); % classes
n = 10; % number of most frequent words to list
m = length(cl); % number of classes
classwords = cell(m,n);
classfreqs = zeros(m,n);
for k = 1:m
    i = strcmp(cl(k),clvar);
    [cfreq,idx] = sort(sum(tdm(i,:),1),'descend');
    cvocab = vocabfull(idx);    
    classwords(k,:) = cvocab(1:n);
    classfreqs(k,:) = cfreq(1:n);
end
classSemantics = {classwords;classfreqs};
% word rank id
wr = {};
for j = 1:n;
    jchar = num2str(j);
    wr = [wr strcat('rank',jchar)];
end
colName = ['Target' wr] ;
classTable = cell2table([cl classwords],'VariableNames',colName);
% classifier
% multinomial naive bayesian classifier on dating classes
mdl = fitcnb(tdm,clvar,'Distribution','mn');
% misclassification error
err = resubLoss(mdl,'LossFun','ClassifErr');
% confusion matrix
labels = resubPredict(mdl);
cmat = confusionmat(clvar,labels);
f2 = plot_confmat(cmat,unique(clvar));
saveFig(strcat(fpath,'/cmat.png'))

run('presskey.m')

% ### misclassification semantics
classmat = [num2cell([1:length(clvar)]') clvar labels];
    classmat(:,4) = num2cell(strcmp(classmat(:,2),classmat(:,3)));
    idx = find(cell2mat(classmat(:,4)) == 0);
misclass = classmat(idx,:);
c = size(misclass,1);
misclasswords = cell(c,n);
misclassfreqs = zeros(c,n);
for i = 1:c    
    doci = cell2mat(misclass(i,1));
    mistdm = tdm(doci,:);
    [temp,idx] = sort(mistdm,'descend');
    temp2 = vocabfull(idx);
    misclassfreqs(i,:) = temp(1:n);
    misclasswords(i,:) = temp2(1:n);
end
misclass = [misclass(:,1:3) misclasswords]; 
colName = ['idDocRow' 'Target' 'Output' wr];
misclassTable = cell2table(misclass,'VariableNames',colName);
disp(classTable)
disp(misclassTable)
%
disp('Press key to continue');
input('');

% ##### explore classification error
% misclassified documents
mcl_idx = [misclass{:,1}]';
mcl_dtm = tdm(mcl_idx,:);
%mcl_dtm = bsxfun(@rdivide,tdm(mcl_idx,:),sum(tdm(mcl_idx,:),2));% relative frequency
% build class dtm without misclassified documents
[~,~,cl_id] = unique(clvar); cl_id = [[classmat{:,1}]' cl_id];
cl_id(mcl_idx,:) = []; % remove misclass

% ## calculate average KL divergence between misclassified and correctly classified
% documents
w = .000001;
mcl_kld = [];
for i = 1:size(mcl_dtm,1)
    tmp = mcl_dtm(i,: )+ w;
    for ii = 1:max(cl_id(:,2))
        cl_tdm = tdm(cl_id(:,2) == ii,:) + w;
        %cl_tdm = bsxfun(@rdivide, cl_tdm, sum(cl_tdm,2));% relative frequency
        kld = KLDiv(cl_tdm,tmp);
        mcl_kld = [mcl_kld; [mean(kld) std(kld) ii misclass{1} i]]; %#ok<AGROW>
        
    end    
end
% plot kl for misclassified documents, 
f3 = figure(3); set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
r = 2; c = ceil(size(misclass,1)/r);% row and columns
for i = 1:size(misclass,1)

% # only for 'chineseTest.mat'
order_id = [3 4 2 1];
era = unique(clvar); era = era(order_id);
% # else
% era = unique(clvar);
x = [1:max(mcl_kld(:,3))]';
% # only for 'chineseTest.mat'
y = mcl_kld(mcl_kld(:,5) == i,1); y = y(order_id);
e = mcl_kld(mcl_kld(:,5) == i,2); e = e(order_id);
% # else
% y = mcl_kld(mcl_kld(:,5) == i,1);
% e = mcl_kld(mcl_kld(:,5) == i,2);

fs = subplot(r,c,i);
h = errorbar(x,y,e);
    h.Color = [0 0 0];
    h.LineStyle = 'None';
    h.LineWidth = 1.5;
set(gca,'xtick',x,'xticklabel',era)   
hold on
% add colors for target (green) and misclass (red)
target = strmatch(misclass(i,2),era);
output = strmatch(misclass(i,3),era);
h0 = scatter(x,y,80,'o','filled'); h0.MarkerFaceColor = [0 0 0];
h1 = scatter(x(target),y(target),80,'o','filled'); h1.MarkerFaceColor = [0 1 0];
h2 = scatter(x(output),y(output),80,'o','filled'); h2.MarkerFaceColor = [1 0 0];
% aesthetics
subaxes = findall(fs,'Type','axes');
set(subaxes,'FontName','Helvetica','FontWeight','Bold','LineWidth',1.5,...
'FontSize',8);
xlabel('Epoch', 'FontSize',14);
ylabel('Bits', 'FontSize',14);
title(['Doc ID: ',num2str(misclass{i})])
end
figSize(f3,15,25);
% saveFig(strcat(fpath,'/mclss_kld.png'))
%% global assessment of feature relevance (mutual informattion based feature selection)
    % Quadratic Programming Feature Selection and MRMR
% - Maximum relevance (maxRel)
% - Minimum redundancy maximum relevance (MRMR)
% - Minimum redundancy (minRed)
% - Quadratic programming feature selection (QPFS)
% - Mutual information quotient (MIQ)
% tdm_norm = normalize_max(tdm); %normalize the data to the [-1,1] range
[~,~,clvar_num] = unique(clvar); % numerical class information
mx_feat = 10;
%[qpfs,maxrel,mrmr,mi_mat,minred,miq] = myQPFS(tdm,clvar_num,mx_feat);
%save('feat_relevance.mat','mrmr')

load('feat_relevance.mat')
dtm_mrmr = tdm(:,mrmr);
cl_dtm_mrmr = dtm_mrmr(cl_id(:,1),:); 
mcl_dtm_mrmr = dtm_mrmr(mcl_idx,:);
avg_feat = zeros(max(cl_id(:,2)),size(dtm_mrmr,2),2);
for i = 1:max(cl_id(:,2))
    avg_feat(i,:,1) = mean(cl_dtm_mrmr(cl_id(:,2) == i,:),1);
    avg_feat(i,:,2) = std(cl_dtm_mrmr(cl_id(:,2) == i,:),1);    
end

hf = figure(1);
val = avg_feat(:,:,1);
subplot(2,1,1), h1 = imagesc(val);
set(gca,'ytick',1:max(cl_id(:,2)),'yticklabel',unique(clvar), ...
    'xtick',1:size(dtm_mrmr,2),'xticklabel',vocabfull(mrmr))
hcl = colorbar;
subplot(2,1,2), h2 = imagesc(mcl_dtm_mrmr);
hcl = colorbar;
tmp = [misclassTable{:,2} misclassTable{:,3}];
l = cell(size(tmp,1),1);
for ii = 1:size(tmp,1); l(ii) = strcat(tmp(ii,1),'>',tmp(ii,2)); end
set(gca,'ytick',1:max(cl_id(:,2)),'yticklabel',l, ...
    'xtick',1:size(dtm_mrmr,2),'xticklabel',vocabfull(mrmr))
colormap(flipud(bone))
subaxes = findall(fs,'Type','axes');
plot_correct
title('Misclassified documents', 'FontSize',14)
xlabel('Character', 'FontSize',14);
ylabel('Class', 'FontSize',14);
subplot(2,1,1),
title('MRMR features', 'FontSize',14)
xlabel('Character', 'FontSize',14);
ylabel('Class', 'FontSize',14);
figSize(hf,20,30)
saveFig(strcat(fpath,'/mclss_mi.png'))
%%



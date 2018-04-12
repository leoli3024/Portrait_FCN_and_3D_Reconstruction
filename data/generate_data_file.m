% generate training data

% the face tracker of the reference image
load('images_tracker/00047.mat')
reftrack = tracker;
refpos = floor(mean(tracker));
[xxc yyc] = meshgrid(1:1800,1:2000);
% normalize the x- and y-channels
xxc = (xxc-600-refpos(1))/600;
yyc = (yyc-600-refpos(2))/800;

maskc = im2double(imread('meanmask.png'));
maskc = padarray(maskc,[600 600]);

% save files into the following folders
files = dir('images_data_crop/*.jpg');

if ~exist('portraitFCN_data','dir')
    mkdir('portraitFCN_data');
end
if ~exist('portraitFCN+_data','dir')
    mkdir('portraitFCN+_data');
end

img = [];
for i=1: length(files)
    imgname = files(i).name
    imgcpy = double(imread(['images_data_crop/' imgname]));
    if size(imgcpy,3)~=3
        imgcpy = repmat(imgcpy(:,:,1),[1 1 3]);
    end
    img = imgcpy;
    img(:,:,1) = (imgcpy(:,:,3) - 104.008)/255;
    img(:,:,2) = (imgcpy(:,:,2) - 116.669)/255;
    img(:,:,3) = (imgcpy(:,:,1) - 122.675)/255;
    
    load(['images_tracker/' imgname(1:end-4) '.mat']);
    destracker = tracker;
    
    if size(tracker,1)==49
        [tform,~,~] = estimateGeometricTransform(double(reftrack)+repmat([600 600],[49 1]),...
        double(tracker)+repmat([600 600],[49 1]),'affine');
        outputView = imref2d(size(xxc));
        warpedxx = imwarp(xxc,tform,'OutputView',outputView);
        warpedyy = imwarp(yyc,tform,'OutputView',outputView);
        warpedmask = imwarp(maskc,tform,'OutputView',outputView);
      
        warpedxx = warpedxx(601:1400,601:1200,:);
        warpedyy = warpedyy(601:1400,601:1200,:);
        warpedmask = warpedmask(601:1400,601:1200,:);
        
        % PortraitFCN data
        save(['portraitFCN_data/' imgname(1:end-4) '.mat'],'img');
        
        % portraitFCN+ data
        imgcpy = img;
        img = zeros(800,600,6);
        img(:,:,1:3) = imgcpy;
        img(:,:,4) = warpedxx;
        img(:,:,5) = warpedyy;
        img(:,:,6) = warpedmask;
        save(['portraitFCN+_data/' imgname(1:end-4) '.mat'],'img');
    else
        % PortraitFCN data
        save(['portraitFCN_data/' imgname(1:end-4) '.mat'],'img');
        
        % portraitFCN+ data
        imgcpy = img;
        img = zeros(800,600,6);
        img(:,:,1:3) = imgcpy;
        save(['portraitFCN+_data/' imgname(1:end-4) '.mat'],'img');
    end
end

% get the training data list

load trainlist.mat;

fid1 = fopen('portraitFCN_datalist.txt','w');
fid2 = fopen('portraitFCN+_datalist.txt','w');
dataroot = pwd;

numimg = 0;
for i=1:length(trainlist)
    
    if exist(['portraitFCN_data/' sprintf('%05d', trainlist(i)) '.mat'],'file')
        outstr = [dataroot '/portraitFCN_data/' sprintf('%05d',trainlist(i)) '.mat\t'  ...
            dataroot '/images_mask/' sprintf('%05d',trainlist(i)) '_mask.mat\n'];
        
        fprintf(fid1,outstr);
        
        outstr = [dataroot '/portraitFCN+_data/' sprintf('%05d',trainlist(i)) '.mat\t'  ...
            dataroot '/images_mask/' sprintf('%05d',trainlist(i)) '_mask.mat\n'];
        
        fprintf(fid2,outstr);
        numimg = numimg + 1;
    end
    
end

fclose(fid1);
fclose(fid2);

disp([num2str(numimg) ' training images are applied!']);


% download data

imgs_names = textread('alldata_urls.txt','%s','delimiter','\n','whitespace','');

data_folder = 'images_data';
if ~exist(data_folder,'dir')
    mkdir(data_folder);
end

if ~exist([data_folder '_crop'],'dir')
   mkdir([data_folder '_crop']) 
end

for i=1:length(imgs_names)
    disp(['download image # ' num2str(i) ' ...']);
    try
        urlwrite(imgs_names{i}(11:end),[data_folder '/' imgs_names{i}(1:9)]);
    end
end

% crop the downloaded images
crop_parms = textread('crop.txt','%s','delimiter','\n','whitespace','');
for i=1:length(crop_parms)
    imgname = crop_parms{i}(1:9);
    crop_xy = str2num(crop_parms{i}(11:end));
    if exist([data_folder '/' imgname],'file')
        try
            img = imread([data_folder '/' imgname]);
            img = img(crop_xy(1)+1:crop_xy(2),crop_xy(3)+1:crop_xy(4),:);
            img = imresize(img,[800 600]);
            imwrite(img,[data_folder '_crop/' imgname]);
        end
    end
end
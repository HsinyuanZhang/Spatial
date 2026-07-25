clear all
clc
%% Load Files 
[fileFeatures,pathFeatures]   = uigetfile('../Database/*.mat','Select Features File');
load(strcat(pathFeatures, fileFeatures));
RowNum = size(features, 1);
ColNum = size(features, 2);

%% Sort Input Feautes based on the time
SFdeltaX       = [];
SFdeltaY       = [];
detectedRows   = [];
detectedCols   = [];
detectedTimes  = [];

for r = 1 : RowNum
    for c = 1 : ColNum
        if ~isempty(features{r, c}.time)
            SFdeltaX          = [SFdeltaX, features{r, c}.X];
            SFdeltaY          = [SFdeltaY, features{r, c}.Y];
            detectedRows      = [detectedRows, r * ones(size(features{r, c}.X)) - 1];
            detectedCols      = [detectedCols, c * ones(size(features{r, c}.X)) - 1];
            detectedTimes     = [detectedTimes, features{r, c}.time];
        end
    end
end

[detectedTimes, timeSortIndex] = sort(detectedTimes);
SFdeltaX                       = SFdeltaX(timeSortIndex);
SFdeltaY                       = SFdeltaY(timeSortIndex);
detectedRows                   = detectedRows(timeSortIndex);
detectedCols                   = detectedCols(timeSortIndex);



%% Parameters
Fs = 20000;
removeScale = 8;

batchSize   = 6 * Fs;
Fs          = 20000;

sampleNum = floor(detectedTimes(end) / Fs) * Fs;

%% Training
batchCntr         = 0;
reset             = 1;
converge          = 0;
validNumArr       = RowNum * ColNum * 2 * 2;
global validMem 

while converge == 0
    startTime     = batchCntr * batchSize;
    batchCntr     = mod(batchCntr + 1, floor(sampleNum / batchSize));
    batchIndex    = (detectedTimes >= startTime) & (detectedTimes < (startTime + batchSize));

    [converge, ~] = Classification(SFdeltaX(batchIndex), SFdeltaY(batchIndex), detectedRows(batchIndex), detectedCols(batchIndex), reset, 1, removeScale, RowNum, ColNum);
    reset         = 0;

    validNumArr   = [validNumArr, sum(sum(validMem == 1))];
end

global deltaXMem
global deltaYMem
global clusters


figure
plot(validNumArr, 'k--o')

%% Clustering
[~, clusterInd] = Classification(SFdeltaX, SFdeltaY, detectedRows, detectedCols, reset, 0, removeScale, RowNum, ColNum);


if strcmp(fileFeatures(1 : 10), 'FeaturesGT')
    save(strcat(pathFeatures, '\','ClustersGT', '_', num2str(removeScale), fileFeatures(11 : end)), 'validNumArr', 'spatialResolution', 'sampleNum', 'validMem', 'deltaXMem', 'deltaXMem', 'clusterInd', 'detectedTimes');
else
    save(strcat(pathFeatures, '\','Clusters_', num2str(ampThreshold), '_', num2str(NEOThreshold),'_', num2str(removeScale),'.mat'), 'removeScale', 'validNumArr', 'spatialResolution', 'sampleNum', 'validMem', 'deltaXMem', 'deltaXMem', 'clusterInd', 'detectedTimes');
end


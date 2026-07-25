clear all
clc

%% Parameters
[file,path]      = uigetfile('../Database/*.h5','File Selector');
recordingFile    = strcat(path, file);
file             = file(1 : end - 4);
numberOfDataSets = 5;

Fs            = 20000;
RowNum        = 32;
ColNum        = 32;
sampleNumber  = h5info(recordingFile).Datasets(3).Dataspace.Size(2);
batchLen      = 512;

bitNum        = 8;
lsbStep       = 4;
wireNum       = 4;

if exist(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'), 'file')==2
  delete(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'));
end
h5create(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'),'/wiredOR',[inf ColNum RowNum],'ChunkSize',[8 8 8])
h5create(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'),'/rawData',[inf ColNum RowNum],'ChunkSize',[8 8 8])

for fileIndex = 0 : (numberOfDataSets - 1)
    recordingFile    = strcat(path, file, num2str(fileIndex), '.h5');
    
    for i = 1 : ceil(sampleNumber / batchLen)
        if i == ceil(sampleNumber / batchLen)
            readLen = sampleNumber - (i - 1) * batchLen;
        else
            readLen = batchLen;
        end
    
        dataInBatch  = h5read(recordingFile,'/recordings', [1, (i - 1) * batchLen + 1], [RowNum * ColNum, readLen]);
        dataInBatch  = int16(round(dataInBatch / lsbStep));
        dataInBatch  = permute(reshape(dataInBatch, ColNum, RowNum, []), [2 1 3]);
    
        dataOutBatch   = WiredOR(double(dataInBatch), wireNum);

        dataOutWiredOr = int16(permute(dataOutBatch, [3 2 1]));
        dataInBatch    = int16(permute(dataInBatch, [3 2 1]));

        h5write(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'), '/wiredOR', dataOutWiredOr, [fileIndex * sampleNumber + (i - 1) * batchLen + 1 1 1], [readLen ColNum RowNum]);
        h5write(strcat(path, '\', num2str(bitNum) , 'b_', num2str(wireNum), 'w/', 'Data.h5'), '/rawData', dataInBatch, [fileIndex * sampleNumber + (i - 1) * batchLen + 1 1 1], [readLen ColNum RowNum]);
    
    
        disp(strcat('Done: ', num2str(100 * i / ceil(sampleNumber / batchLen)), '%'))
    end

end


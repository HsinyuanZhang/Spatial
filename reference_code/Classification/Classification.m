function [converge, clusterIndex] = Classification(deltaX, deltaY, row, column, reset, training, removeScale, rowNum, colNum)

global deltaXMem
global deltaYMem
global validMem
global counterMem
global clusters;

clusterIndex = zeros(numel(deltaX), 1) - 1;
if reset == 1
    deltaXMem = zeros(rowNum * 2, colNum * 2);
    deltaYMem = zeros(rowNum * 2, colNum * 2);
    validMem  = ones(rowNum * 2, colNum * 2);
    counterMem= zeros(rowNum * 2, colNum * 2);
    clusters  = zeros(rowNum * 2, colNum * 2);
    validNum  = rowNum * 2 * colNum * 2; 
end
validNum_r = sum(sum(validMem));


if training == 1
    counterMem(:, :) = 0;
end

rowStart    = 2 * row - 2;
columnStart = 2 * column + mod(row, 2) - 2;

locationX   = ((2 * column + mod(row, 2)) * 2 ^ 5) + deltaX;
locationY   = ((2 * row) * 2 ^ 5) + deltaY;

for i = 1 : numel(deltaX)
   found = 0;
    minDist = Inf;
    for r = 0 : 4
        for c = 0 : 4
            rAddr     = rowStart(i) + r;
            cAddr     = columnStart(i) + c;

            if rAddr < 0 || rAddr >= 2 * rowNum || cAddr < 0 || cAddr >= 2 * colNum
                continue;
            end

            locationXRead = ((cAddr) * 2 ^ 5) + deltaXMem(rAddr + 1, cAddr + 1);
            locationYRead = ((rAddr) * 2 ^ 5) + deltaYMem(rAddr + 1, cAddr + 1);

            locationDist  = sqrt((locationX(i) - locationXRead)^2 + (locationY(i) - locationYRead)^2);
            if locationDist < minDist && validMem(rAddr + 1, cAddr + 1) == 1
                minDist          = locationDist;
                locationXReadMin = locationXRead;
                locationYReadMin = locationYRead;
                rowMin           = rAddr;
                colMin           = cAddr;
                found            = 1;
            end
        end
    end
    if found == 1
        clusterIndex(i) = clusters(rowMin + 1, colMin + 1);
       
        locationXNew = round((locationXReadMin * 15 + locationX(i)) / 16);
        locationYNew = round((locationYReadMin * 15 + locationY(i)) / 16);
    
        rowNew       = round(locationYNew / 32);
        colNew       = round(locationXNew / 32);
    
        deltaXNew    = locationXNew - colNew * 32;
        deltaYNew    = locationYNew - rowNew * 32;
            
        if (rowMin == rowNew && colMin == colNew)
            deltaXMem(rowMin + 1, colMin + 1) = deltaXNew;
            deltaYMem(rowMin + 1, colMin + 1) = deltaYNew;
    
            counterMem(rowMin + 1, colMin + 1) = counterMem(rowMin + 1, colMin + 1) + 1;
        else

            validMem(rowMin + 1, colMin + 1) = 0;
            deltaXMem(rowMin + 1, colMin + 1) = 0;
            deltaYMem(rowMin + 1, colMin + 1) = 0;
            
    
            validMem(rowNew + 1, colNew + 1) = 1;
            clusters(rowNew + 1, colNew + 1) = clusters(rowMin + 1, colMin + 1);
            clusters(rowMin + 1, colMin + 1) = 0;
    
            deltaXMem(rowNew + 1, colNew + 1) = deltaXNew;
            deltaYMem(rowNew + 1, colNew + 1) = deltaYNew;
    
            counterMem(rowNew + 1, colNew + 1) = counterMem(rowNew + 1, colNew + 1) + counterMem(rowMin + 1, colMin + 1) + 1;
            counterMem(rowMin + 1, colMin + 1) = 0;
        end
    end
end

converge = 0;
if training == 1
    counterAvg = ceil(numel(deltaX) / validNum_r / removeScale);
    validMem(counterMem < counterAvg) = 0;
    validNum = sum(sum(validMem));

    if validNum == validNum_r
        converge = 1;
        
        for r = 1 : rowNum * 2 - 1
            for c = 1 : colNum * 2 - 1
                maxCount = floor(max(reshape(counterMem(r : r + 1, c : c + 1), 1, [])) / removeScale);
                validMem(r : r + 1, c : c + 1) = validMem(r : r + 1, c : c + 1) .* (counterMem(r : r + 1, c : c + 1) > maxCount);
                clusters(r, c) = r * 2 * colNum + c;
            end
        end
        validNum = sum(sum(validMem));
        clusters = clusters .* validMem;
    end
end

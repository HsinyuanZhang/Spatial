function [NeighborElecsRow, NeighborElecsCol, NeighborElecsCos, NeighborElecsSin, WrongPosition] = FindNeighborElec(mainElecRow, mainElecCol, RowNum, ColNum)

NeighborElecsLocX  = [-60, 60, -30, 30, -30, 30];
NeighborElecsLocY  = [0, 0, 60, 60, -60, -60];
NeighborElecsLoc   = NeighborElecsLocX + 1i* NeighborElecsLocY;
NeighborElecsAngle = angle(NeighborElecsLoc);
NeighborElecsCos   = cos(NeighborElecsAngle); 
NeighborElecsSin   = sin(NeighborElecsAngle); 

NeighborElecsRow(1) = mainElecRow;
NeighborElecsCol(1) = mainElecCol - 1;
NeighborElecsRow(2) = mainElecRow;
NeighborElecsCol(2) = mainElecCol + 1;
if mod(mainElecRow, 2) == 1
    NeighborElecsRow(3) = mainElecRow + 1;
    NeighborElecsCol(3) = mainElecCol - 1;

    NeighborElecsRow(4) = mainElecRow + 1;
    NeighborElecsCol(4) = mainElecCol;

    NeighborElecsRow(5) = mainElecRow - 1;
    NeighborElecsCol(5) = mainElecCol - 1;

    NeighborElecsRow(6) = mainElecRow - 1;
    NeighborElecsCol(6) = mainElecCol;
else
    NeighborElecsRow(3) = mainElecRow + 1;
    NeighborElecsCol(3) = mainElecCol;

    NeighborElecsRow(4) = mainElecRow + 1;
    NeighborElecsCol(4) = mainElecCol + 1;

    NeighborElecsRow(5) = mainElecRow - 1;
    NeighborElecsCol(5) = mainElecCol;

    NeighborElecsRow(6) = mainElecRow - 1;
    NeighborElecsCol(6) = mainElecCol + 1;
end

WrongPosition = ~(NeighborElecsRow < 1 | NeighborElecsRow > RowNum | NeighborElecsCol < 1 | NeighborElecsCol > ColNum);
NeighborElecsCos(~WrongPosition) = 0;
NeighborElecsSin(~WrongPosition) = 0;
NeighborElecsRow(~WrongPosition) = mainElecRow;
NeighborElecsCol(~WrongPosition) = mainElecCol;


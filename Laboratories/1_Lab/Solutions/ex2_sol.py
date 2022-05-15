import sys

class BusRecord:
    def __init__(self, busId, lineId, x, y, t):
        self.busId = busId
        self.lineId = lineId
        self.x = x
        self.y = y
        self.t = t
        
def loadAllRecords(fName):
    
    try:
        lRecords = []
        with open(fName) as f:
            for line in f:
                busId, lineId, x, y, t = line.split()
                record = BusRecord(busId, lineId, int(x), int(y), int(t))
                lRecords.append(record)
        return lRecords
    except:
        raise # If we do not provide an exception, the current exception is propagated

def euclidean_distance(r1, r2):
    return ((r1.x-r2.x)**2 + (r1.y-r2.y)**2)**0.5

def computeBusDistanceTime(lRecords, busId):
    busRecords = sorted([i for i in lRecords if i.busId == busId], key = lambda x: x.t)
    if len(busRecords) == 0:
        return None, None
    totDist = 0.0
    for prev_record, curr_record in zip(busRecords[:-1], busRecords[1:]):
        totDist += euclidean_distance(curr_record, prev_record)
    totTime = busRecords[-1].t - busRecords[0].t
    return totDist, totTime

def computeLineAvgSpeed(lRecords, lineId):
    
    lRecordsFiltered = [i for i in lRecords if i.lineId == lineId]
    busSet = set([i.busId for i in lRecordsFiltered])
    if len(busSet) == 0:
        return 0.0
    totDist = 0.0
    totTime = 0.0
    for busId in busSet:
        d, t = computeBusDistanceTime(lRecordsFiltered, busId)
        totDist += d
        totTime += t
    return totDist / totTime

if __name__ == '__main__':

    lRecords = loadAllRecords(sys.argv[1])
    if sys.argv[2] == '-b':
        print('%s - Total Distance:' % sys.argv[3], computeBusDistanceTime(lRecords, sys.argv[3])[0])
    elif sys.argv[2] == '-l':
        print('%s - Avg Speed:' % sys.argv[3], computeLineAvgSpeed(lRecords, sys.argv[3]))
    else:
        raise KeyError()


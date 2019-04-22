import math
import numpy as np

THRESH_HEAT = 0.05
THRESH_VECTOR_SCORE = 0.05
THRESH_VECTOR_CNT1 = 6 # How many out of the STEP_PAF samples should be Good
THRESH_PART_CNT = 4
THRESH_HUMAN_SCORE = 0.4
NUM_PART = 14
NUM_HEATMAP = NUM_PART + 1
STEP_PAF = 10
NUM_BC_PAIRS = 13
from tf_pose.common import BC_pairs as BC_PAIRS

def roundpaf(num):
    return int(num + 0.5)

class Peak:
    def __init__(self, peakGlobalId, peakX, peakY, paekScore):
        self._peakGlobalId = peakGlobalId
        self._peakX = peakX
        self._peakY = peakY
        self._paekScore = paekScore


class VectorXY:
    def __init__(self, x, y):
        self._x = x
        self._y = y

class ConnectionCandidate:
    def __init__(self, peak_id_0, peak_id_1, score, etc):
        self._peak_id_0 = peak_id_0
        self._peak_id_1 = peak_id_1
        self._score = score
        self._etc = etc

    def __lt__(self, other):
        return self._score < other._score


class Connection:
    def __init__(self, peakListId0, peakListId1, score, peakGlobalId0, peakGlobalId1):
        self._peakGlobalId0 = peakGlobalId0
        self._peakGlobalId1 = peakGlobalId1
        self._score = score
        self._peakListId0 = peakListId0
        self._peakListId1 = peakListId1


def estimate_paf(peaks, heatMat, pafMat):
    rows = heatMat.shape[0]
    cols = heatMat.shape[1]
    all_peaks = [[] for i in range(NUM_PART)]
    peak_count = 0
    for part_id in range(NUM_PART):
        for y in range(rows):
            for x in range(cols):
                if(peaks[y,x, part_id] > THRESH_HEAT):
                    all_peaks[part_id] += [Peak(peak_count, x, y, heatMat[y,x, part_id])]
                    peak_count += 1

    flat_peaks = list(np.concatenate(all_peaks))

    bestValidConnections = [[] for i in range(NUM_BC_PAIRS)] # class Connection list

    for pair_id in range(NUM_BC_PAIRS):

        connectionCandidates = [] # class ConnectionCandidate list
        peak_a_list = all_peaks[BC_PAIRS[pair_id][0]]
        peak_b_list = all_peaks[BC_PAIRS[pair_id][1]]

        if peak_a_list == [] or peak_b_list == []:
            continue

        for peak_0_id in range(len(peak_a_list)):
            for peak_1_id in range (len (peak_b_list)):
                x_diff = peak_b_list[peak_1_id]._peakX - peak_a_list[peak_0_id]._peakX
                y_diff = peak_b_list[peak_1_id]._peakY - peak_a_list[peak_0_id]._peakY

                norm = math.sqrt (x_diff** 2 + y_diff ** 2)
                if norm < 1e-12:
                    continue
                direction_vec = VectorXY(x_diff/ norm, y_diff/ norm)

                # compute PAF
                scores = 0.0
                numValidVectorFraction = 0
                for step in range(STEP_PAF):
                    step_location_x = roundpaf(peak_a_list[peak_0_id]._peakX + step * x_diff / float(STEP_PAF))
                    step_location_y = roundpaf(peak_a_list[peak_0_id]._peakY + step * y_diff / float (STEP_PAF))
                    step_value_x = pafMat[step_location_y, step_location_x, 2*pair_id]
                    step_value_y = pafMat[step_location_y, step_location_x, 2*pair_id + 1]
                    score = direction_vec._x * step_value_x  + direction_vec._y*step_value_y
                    scores += score
                    if score > THRESH_VECTOR_SCORE:
                        numValidVectorFraction += 1

                totalScoreCriterion = scores / STEP_PAF + min(0.0, 0.5 * rows / norm -1.0)

                if numValidVectorFraction > THRESH_VECTOR_CNT1 and totalScoreCriterion > 0:
                    candidate = ConnectionCandidate(peak_id_0=peak_0_id,
                                                    peak_id_1=peak_1_id,
                                                    score=totalScoreCriterion,
                                                    etc=totalScoreCriterion + peak_a_list[peak_0_id]._paekScore +
                                                                                peak_b_list[peak_1_id]._paekScore)
                    connectionCandidates += [(candidate)]


        # save only best connection for each peak
        connectionCandidates.sort(reverse=True)
        for candidate in connectionCandidates:
            one_of_peaks_assigned = False
            for connection in bestValidConnections[pair_id]:
                if connection._peakListId0 == candidate._peak_id_0 or connection._peakListId1 == candidate._peak_id_1:
                    one_of_peaks_assigned = True
                    break;
            if one_of_peaks_assigned:
                continue
            bestValidConnections[pair_id] += [Connection(peakListId0=candidate._peak_id_0,
                                                         peakListId1=candidate._peak_id_1,
                                                         score=candidate._score,
                                                         peakGlobalId0=peak_a_list[candidate._peak_id_0]._peakGlobalId,
                                                         peakGlobalId1=peak_b_list[candidate._peak_id_1]._peakGlobalId
                                              )]

    subset = []
    for pair_id in range(NUM_BC_PAIRS):
        for connection in bestValidConnections[pair_id]:
            num_intersectiong_humans = 0
            subset_idx0 = 0
            subset_idx1 = 0
            for subset_id, subset_cell in enumerate(subset):
                if subset_cell[BC_PAIRS[pair_id][0]] == connection._peakGlobalId0 or \
                        subset_cell[BC_PAIRS[pair_id][1]] == connection._peakGlobalId1:
                    if num_intersectiong_humans == 0:
                        subset_idx0 = subset_id
                    if num_intersectiong_humans == 1:
                        subset_idx1 = subset_id
                    num_intersectiong_humans += 1

            if num_intersectiong_humans ==1 : # if not already there, add the other peak to this human
                if subset[subset_idx0][BC_PAIRS[pair_id][1]] != connection._peakGlobalId1:
                    subset[subset_idx0][BC_PAIRS[pair_id][1]] = connection._peakGlobalId1
                    subset[subset_idx0][NUM_PART] += flat_peaks[connection._peakGlobalId1]._paekScore + connection._score
                    subset[subset_idx0][NUM_HEATMAP] += 1

            elif num_intersectiong_humans == 2:
                membership = 0
                for subset_part_id in range(NUM_PART):
                    if subset[subset_idx0][subset_part_id] > 0 and subset[subset_idx1][subset_part_id] > 0:
                        membership = 2
                        break
                if membership == 0:
                    for subset_part_id in range (NUM_PART):
                        subset[subset_idx0][subset_part_id] += subset[subset_idx1][subset_part_id] + 1 # +1 for the -1 intialization?
                    subset[subset_idx0][NUM_HEATMAP] += subset[subset_idx1][NUM_HEATMAP]
                    subset[subset_idx0][NUM_PART] += subset[subset_idx1][NUM_PART]

                else:
                    subset[subset_idx0][BC_PAIRS[pair_id][1]] = connection._peakGlobalId2
                    subset[subset_idx0][NUM_HEATMAP] += 1
                    subset[subset_idx0][NUM_PART] += flat_peaks[connection._peakGlobalId1]._paekScore

                subset[subset_idx0][NUM_PART] += connection._score

            elif num_intersectiong_humans == 0 and pair_id < NUM_BC_PAIRS:
                row = [-1]*(NUM_PART + 2)
                row[BC_PAIRS[pair_id][0]] = connection._peakGlobalId0
                row[BC_PAIRS[pair_id][1]] = connection._peakGlobalId1
                row[NUM_PART] = flat_peaks[connection._peakGlobalId0]._paekScore + \
                                flat_peaks[connection._peakGlobalId1]._paekScore + \
                                connection._score
                row[NUM_HEATMAP] = 2
                subset += [row]

    new_subset = []
    for instance in subset:
        if (instance[NUM_HEATMAP] < THRESH_PART_CNT or (instance[NUM_PART] / instance[NUM_HEATMAP]) < THRESH_HUMAN_SCORE):
            continue
        else:
            new_subset.append(instance)

    return new_subset, flat_peaks
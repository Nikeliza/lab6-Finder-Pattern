import cv2
import math
import numpy as np
import time

class finder_pattern_detector:
    def __init__(self):
        self.centers = []
        self.module_size = []
        self.path_input_image = ''
        self.input_image = None
        self.image_black_white = None
        self.pram = None

    def read_image(self, path):
        self.path_input_image = path
        self.input_image = cv2.imread(self.path_input_image)
        self.image_black_white = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        self.image_black_white = cv2.adaptiveThreshold(self.image_black_white, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
        # apply GaussianBlur to smooth image
        self.image_black_white = cv2.GaussianBlur(self.image_black_white, (5, 3), 1)
        self.image_black_white = cv2.inRange(self.image_black_white, (0), (150))
        self.image_black_white = 255 - self.image_black_white
        contours0, hierarchy = cv2.findContours(self.image_black_white.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # перебираем все найденные контуры в цикле
        max_angle = 0
        max_squer = 0
        for cnt in contours0:
            rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            area = int(rect[1][0] * rect[1][1])  # вычисление площади
            if area > 1000:
                if area > max_squer:
                    max_squer = area
                    max_angle = box
        Xs = [i[0] for i in max_angle]
        Ys = [i[1] for i in max_angle]
        x1 = min(Xs)
        if x1 < 0:
            x1 = 0
        x2 = max(Xs)
        if x2 > self.image_black_white.shape[1]:
            x2 = self.image_black_white.shape[1]
        y1 = min(Ys)
        if y1 < 0:
            y1 = 0
        y2 = max(Ys)
        if y2 > self.image_black_white.shape[0]:
            y2 = self.image_black_white.shape[0]
        self.pram = [[y1, y2], [x1, x2]]

    def detector(self):
        found = self._find()
        if found:
            self.draw_finder_patterns()
        else:
            self.pram = [[0, self.image_black_white.shape[0]], [1, self.image_black_white.shape[1]]]
            found = self._find()
            if found:
                self.draw_finder_patterns()
        return found

    def create_result_image(self):
        pathResultImage = "result//"
        pathResultImage += self.path_input_image[0:len(self.path_input_image) - 4] + '_result1.jpg'
        cv2.imwrite(pathResultImage, self.input_image)

    def _find(self):
        self.centers = []
        self.module_size = []
        skip_rows = 5
        counterState = [0, 0, 0, 0, 0]
        currentState = 0
        for row in range(self.pram[0][0] + skip_rows - 1, self.pram[0][1], skip_rows):
            counterState = [0, 0, 0, 0, 0]
            currentState = 0
            ptr = self.image_black_white[row, :]
            for col in range(self.pram[1][0], self.pram[1][1]):
                if ptr[col] < 128:# черный пиксель
                    if (currentState & 1) == 1:
                        currentState += 1
                    counterState[currentState] += 1
                else: # белый пиксель
                    if (currentState & 1) == 1: #если 1 или 3 положение
                        counterState[currentState] += 1
                    else:
                        if currentState == 4:
                            if self._check_ratio(counterState):
                                #this is where we do some more checks
                                res = self._handle_possible_center(counterState, row, col)
                            else:
                                currentState = 3
                                counterState[0] = counterState[2]
                                counterState[1] = counterState[3]
                                counterState[2] = counterState[4]
                                counterState[3] = 1
                                counterState[4] = 0
                                continue
                            currentState = 0
                            counterState = [0, 0, 0, 0, 0]
                        else:
                            currentState += 1
                            counterState[currentState] += 1
        return len(self.centers) > 0

    def draw_finder_patterns(self):
        if (len(self.centers) == 0):
            return

        for i in range(len(self.centers)):
            pt = self.centers[i]
            diff = self.module_size[i] * 3.5
            point1 = [pt[0] - diff, pt[1] - diff]
            if point1[0] < 0:
                point1[0] = 0
            if point1[1] < 0:
                point1[1] = 0
            point2 = [pt[0] + diff, pt[1] + diff]
            if abs(point1[0] - point2[0]) > self.image_black_white.shape[0] / 100 and abs(point1[1] - point2[1]) > self.image_black_white.shape[1] / 100:
                cv2.rectangle(self.input_image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 0, 255), 3)

    def _check_ratio(self, state_count):
        totalWidth = 0
        for i in range(5):
            if state_count[i] == 0:
                return False
            totalWidth += state_count[i]

        if totalWidth < 7 and (totalWidth > min(self.image_black_white.shape[0] / 100, self.image_black_white.shape[1] / 100)):
            return False
        width = round(totalWidth / 7.0)
        dispersion = width / 2

        a = (abs(width - (state_count[0])) < dispersion) and (abs(width - (state_count[1])) < dispersion) and \
               (abs(3 * width - (state_count[2])) < 3 * dispersion) and (abs(width - (state_count[3])) < dispersion) and \
                (abs(width - (state_count[4])) < dispersion)
        return a

    def _handle_possible_center(self, state_count, row, col):
        totalState = sum(state_count)
        if totalState > self.image_black_white.shape[0] / 100:
            centerCol = self._get_center(state_count, col)
            centerRow = self._check_vertical(row, centerCol, state_count[2], totalState)
            if centerRow == -1.0:
                return False
            new_center = [centerCol, centerRow]
            if new_center not in self.centers:
                centerCol = self._check_horizontal(centerRow, centerCol, state_count[2], totalState)
                if centerCol == -1.0:
                    return False
                new_center = [centerCol, centerRow]
                if new_center not in self.centers:
                    if not self._check_diagonal(centerRow, centerCol, state_count[2], totalState):
                        return False

                    new_center = [centerCol, centerRow]

                    newModuleSize = totalState / 7.0
                    found = False
                    index = 0
                    for  point in self.centers:
                        diff = [0, 0]
                        diff[0] = point[0] - new_center[0]
                        diff[1] = point[1] - new_center[1]
                        distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

                        if distance < 10:
                            point[0] = point[0] + new_center[0]
                            point[1] = point[1] + new_center[1]
                            point[0] /= 2.0
                            point[1] /= 2.0
                            self.module_size[index] = (self.module_size[index] + newModuleSize) / 2.0
                            found = True
                            break
                        index += 1

                    if not found and new_center not in self.centers:
                        self.centers.append(new_center)
                        self.module_size.append(newModuleSize)

        return False


    def _check_diagonal(self, center_row, center_col, max_count, state_count_total):
        state_count = [0, 0, 0, 0, 0]
        i = 0
        center_row = int(center_row) + 1
        center_col = int(center_col) + 1
        while center_row >= i and center_col >= 1 and self.image_black_white[center_row - i, center_col - i] < 128:
            state_count[2] += 1
            i += 1
            if center_row < i or center_col < i:
                return False

        while center_row >= i and center_col >= i and self.image_black_white[center_row - i, center_col - i] >= 128 and state_count[1] <= max_count:
            state_count[1] += 1
            i += 1
            if center_row < i or center_col < i or state_count[1] > max_count:
                return False

        while center_row >= i and center_col >= i and self.image_black_white[center_row - i, center_col - i] < 128 and state_count[0] <= max_count:
            state_count[0] += 1
            i += 1
            if state_count[0] > max_count:
                return False

        i = 1
        while center_row + i < self.image_black_white.shape[0] and center_col + i < self.image_black_white.shape[1] and self.image_black_white[center_row + i, center_col + i] < 128:
            state_count[2] += 1
            i += 1
            if center_row + i >= self.image_black_white.shape[0] or center_col + i >= self.image_black_white.shape[1]:
                return False

        while center_row + i < self.image_black_white.shape[0] and center_col + i < self.image_black_white.shape[1] and self.image_black_white[center_row + i, center_col + i] >= 128 and state_count[3] < max_count:
            state_count[3] += 1
            i += 1
            if center_row + i >= self.image_black_white.shape[0] or center_col + i >= self.image_black_white.shape[1] or state_count[3] > max_count:
                return False

        while center_row + i < self.image_black_white.shape[0] and center_col + i < self.image_black_white.shape[1] and self.image_black_white[center_row + i, center_col + i] < 128 and state_count[4] < max_count:
            state_count[4] += 1
            i += 1
            if state_count[4] > max_count:
                return False

        new_state_count_total = 0
        for j in range(5):
            new_state_count_total += state_count[j]
        a = (abs(state_count_total - new_state_count_total) < 2 * state_count_total)
        b = self._check_ratio(state_count)
        return a and b

    def _check_vertical(self, start_row, center_col, central_count, state_count_total):
        counter_state = [0, 0, 0, 0, 0]
        row = start_row
        center_col = int(center_col)
        while row >= 0 and self.image_black_white[row, center_col] < 128:
            counter_state[2] += 1
            row -= 1
        if row < 0:
            return -1

        while row >= 0 and self.image_black_white[row, center_col] >= 128 and counter_state[1] < central_count:
            counter_state[1] += 1
            row -= 1
        if row < 0 or counter_state[1] >= central_count:
            return  -1

        while row >= 0 and self.image_black_white[row, center_col] < 128 and counter_state[0] < central_count:
            counter_state[0] += 1
            row -= 1
        if counter_state[0] >= central_count:
            return -1

        row = start_row + 1
        while row < self.image_black_white.shape[0] and self.image_black_white[row, center_col] < 128:
            counter_state[2] += 1
            row += 1
        if row == self.image_black_white.shape[0]:
            return -1

        while row < self.image_black_white.shape[0] and self.image_black_white[row, center_col] >= 128 and counter_state[3] < central_count:
            counter_state[3] += 1
            row += 1
        if row == self.image_black_white.shape[0] or counter_state[3] >= central_count:
            return -1

        while row < self.image_black_white.shape[0] and self.image_black_white[row, center_col] < 128 and counter_state[4] < central_count:
            counter_state[4] += 1
            row += 1
        if counter_state[4] >= central_count:
            return  -1

        counter_state_total = sum(counter_state)

        if 5 * abs(counter_state_total - state_count_total) >= 2 * state_count_total:
            return -1

        center = self._get_center(counter_state, row)
        if self._check_ratio(counter_state):
            return center
        else:
            return -1

    def _check_horizontal(self, center_row, start_col, center_count, state_count_total):
        counter_state = [0, 0, 0, 0, 0]
        col = int(start_col)
        center_row = int(center_row)
        ptr = self.image_black_white[center_row,:]
        while col >= 0 and ptr[col] < 128:
            counter_state[2] += 1
            col -= 1

        if col < 0:
            return -1

        while col >= 0 and ptr[col] >= 128 and counter_state[1] < center_count:
            counter_state[1] += 1
            col -= 1

        if col < 0 or counter_state[1] == center_count:
            return -1

        while col >= 0 and ptr[col] < 128 and counter_state[0] < center_count:
            counter_state[0] += 1
            col -= 1

        if counter_state[0] == center_count:
            return -1

        col = int(start_col) + 1
        while col < self.image_black_white.shape[1] and ptr[col] < 128:
            counter_state[2] += 1
            col += 1

        if col == self.image_black_white.shape[1]:
            return -1

        while col < self.image_black_white.shape[1] and ptr[col] >= 128 and counter_state[3] < center_count:
            counter_state[3] += 1
            col += 1
        if col == self.image_black_white.shape[1] or counter_state[3] == center_count:
            return -1
        while col < self.image_black_white.shape[1] and ptr[col] < 128 and counter_state[4] < center_count:
            counter_state[4] += 1
            col += 1

        if counter_state[4] >= center_count:
            return -1

        counter_state_total = 0
        for i in range(5):
            counter_state_total += counter_state[i]

        if 5 * abs(counter_state_total - state_count_total) >= state_count_total:
            return -1

        center = self._get_center(counter_state, col)
        if self._check_ratio(counter_state):
            return center
        else:
            return -1

    def _get_center(self, state_count, end):
        return end - state_count[4] - state_count[3] - state_count[2] / 2.0

object = finder_pattern_detector()

for i in range(48, 151):
    time1 = time.time()
    print(i, end=' ')
    if i < 10:
        object.read_image("image//TestSet3//000" + str(i) + ".jpg")
    elif i < 100:
        object.read_image("image//TestSet3//00" + str(i) + ".jpg")
    else:
        object.read_image("image//TestSet3//0" + str(i) + ".jpg")
    object.detector()
    object.create_result_image()
    print(time1- time.time())

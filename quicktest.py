import random

class Point:

    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def get_vals(self):
        return (self.x, self.y, self.z)


class CustomIterClass:


    def __init__(self, n_points=10):

        self.point_list: list[Point] = []

        for _ in range(n_points):

            x = random.randint(0,9)
            y = random.randint(0,9)
            z = random.randint(0,9)

            self.point_list.append(Point(x,y,z))
        
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx == len(self.point_list):
            raise StopIteration
        else:
            self.iter_idx += 1
            return self.point_list[self.iter_idx-1].get_vals()
    

if __name__ == "__main__":
    
    print("Main process")

    cst = CustomIterClass(n_points=5)

    for x,y,z in cst:

        print(f"x = {x}, y = {y}, z = {z}")




from simple_playgrounds.playgrounds.empty import SingleRoom, ConnectedRooms2D, LinearRooms
from simple_playgrounds.entities.scene_elements import *
from simple_playgrounds.entities.field import Field

from simple_playgrounds.utils import PositionAreaSampler, Trajectory
from simple_playgrounds.playgrounds.playground import PlaygroundRegister

import random

@PlaygroundRegister.register('spg_endgoal_cue')
class EndgoalRoomCue(SingleRoom):

    def __init__(self):

        # wall_texture = UniqueRandomTilesTexture(n_colors=4, delta_uniform=15, color_min=(50, 50, 50),
        #                                         color_max=(150, 150, 150), radius=100)

        super().__init__(size = (200, 200))#, wall_texture = wall_texture)

        # Starting area of the agent
        area_center, _ = self.area_rooms[(0,0)]
        area_start = PositionAreaSampler(center=area_center, area_shape='rectangle', width_length=(100,100))
        self.agent_starting_area = area_start

        # Obstacles
        obstacle_1 = Basic((60, 30, 0.34), default_config_key='pentagon')
        self.add_scene_element(obstacle_1)

        obstacle_2 = Basic((130, 150, 1.7), default_config_key='rectangle')
        self.add_scene_element(obstacle_2)

        obstacle_3 = Basic((40, 140, 0.4), default_config_key='square')
        self.add_scene_element(obstacle_3)

        obstacle_4 = Basic((160, 60, 0), physical_shape = 'triangle', radius = 14, texture = (150, 200, 200))
        self.add_scene_element(obstacle_4)

        self.goal = None
        self.cue = None

        self.goal_locations = ( (20, 20, 0), (180, 20, 0), (180, 180, 0), (20, 180, 0) )
        self.cue_colors = ( (50, 50, 50), (50, 250, 50), (250, 50, 50), (50, 50, 250))

        self.set_goal()
        self.time_limit = 2000
        self.time_limit_reached_reward = -20


    def set_goal(self):

        index_goal = random.randint(0, 3)
        loc = self.goal_locations[index_goal]
        col = self.cue_colors[index_goal]

        if self.goal is not None:
            self.remove_scene_element(self.goal)
            self.remove_scene_element(self.cue)

        self.cue = Basic((100, 100, 0), physical_shape = 'circle', radius = 10, texture = col, is_temporary_entity = True)
        self.add_scene_element(self.cue)

        self.goal = GoalZone(loc, reward = 10, is_temporary_entity = True)
        self.add_scene_element(self.goal)

        for i in range(4):
            if i != index_goal:
                loc = self.goal_locations[i]
                other_goal = DeathZone(loc, reward=-5, is_temporary_entity=True)
                self.add_scene_element(other_goal)

    def reset(self):
        super().reset()

        self.set_goal()


@PlaygroundRegister.register('spg_endgoal_9rooms')
class Endgoal9Rooms(ConnectedRooms2D):

    def __init__(self):

        super().__init__(size = (600, 600), n_rooms=(3,3), wall_type='colorful')

        # Starting area of the agent
        area_start = PositionAreaSampler(center=(300, 300), area_shape='rectangle', width_length=(600, 600))
        self.agent_starting_area = area_start

        # invisible endzone at one corner of the game
        invisible_endzone = GoalZone((20, 20, 0), reward = 10)
        self.add_scene_element(invisible_endzone)

        self.time_limit = 5000

@PlaygroundRegister.register('spg_dispenser_6rooms')
class DispenserEnv(ConnectedRooms2D):

    def __init__(self):

        super().__init__(size = (300, 200), n_rooms=(3, 2))#, wall_type='colorful')


        self.assign_areas()
        self.place_scene_elements()

        self.time_limit = 2000

    def assign_areas(self):
        list_room_coordinates = [room_coord for room_coord in self.area_rooms]
        random.shuffle(list_room_coordinates)

        # Starting area of the agent
        area_start_center, area_start_shape = self.area_rooms[list_room_coordinates.pop()]
        area_start = PositionAreaSampler(center=area_start_center, area_shape='rectangle',
                                         width_length=area_start_shape)
        self.agent_starting_area = area_start

        # invisible endzone at one corner of the game
        dispenser_center, dispenser_shape = self.area_rooms[list_room_coordinates.pop()]
        prod_center, prod_shape = self.area_rooms[list_room_coordinates.pop()]


        self.area_prod = PositionAreaSampler(center=prod_center, area_shape='rectangle', width_length=prod_shape)
        self.area_disp = PositionAreaSampler(center=dispenser_center, area_shape='rectangle', width_length=dispenser_shape)

    def place_scene_elements(self):


        dispenser = Dispenser(self.area_disp, production_area = self.area_prod, entity_produced=Candy, radius = 10, is_temporary_entity = True, allow_overlapping = False )
        self.add_scene_element(dispenser)


    def reset(self):
        self.assign_areas()
        for agent in self.agents:
            agent.initial_position = self.agent_starting_area
        super().reset()
        self.place_scene_elements()


@PlaygroundRegister.register('spg_coinmaster')
class CoinMaster(SingleRoom):

    def __init__(self):

        super().__init__(size = (200, 200), wall_type='dark')

        # Starting area of the agent
        area_center, _ = self.area_rooms[(0, 0)]
        area_start = PositionAreaSampler(center=area_center, area_shape='rectangle', width_length=(100, 100))
        self.agent_starting_area = area_start

        self.vending_machine = VendingMachine(area_start, allow_overlapping = False)
        self.add_scene_element(self.vending_machine)

        self.field = Field(Coin, area_start, limit=5, total_limit=1000, entity_produced_params={'graspable':True, 'reward':1,
                                                                                                'allow_overlapping': False})
        self.add_scene_element(self.field)

        self.time_limit = 1000

    def _fields_produce(self):

        super()._fields_produce()
        self.vending_machine.accepted_coins = self.field.produced_entities





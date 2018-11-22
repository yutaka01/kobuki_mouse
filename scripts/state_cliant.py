#! /usr/bin/env python

from gazebo_msgs.srv import GetModelState
import rospy , numpy

class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class Tutorial:

    _blockListDict = {
        'r1': Block('r1', ''),
        'r2': Block('r2', ''),
        'r3': Block('r3', ''),
        'r4': Block('r4', ''),
        'r5': Block('r5', ''),
    }

    def show_gazebo_models(self):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            for block in self._blockListDict.itervalues():
                
                blockName = str(block._name)
                resp_coordinates = model_coordinates(blockName, block._relative_entity_name)
                print '\n'
                print 'Status.success = ', resp_coordinates.success
                print(blockName)
                print("Cube " + str(block._name))
                print("x = " + str(resp_coordinates.pose.position.x))
                print("y = " + str(resp_coordinates.pose.position.y))
                pts.append([resp_coordinates.pose.position.x, resp_coordinates.pose.position.y])
            print(pts)
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))


if __name__ == '__main__':
    tuto = Tutorial()
    tuto.show_gazebo_models()
            

import glob
import os
import sys
import numpy
import numpy as np
import pickle


class TreeNode:
    def __init__(self, name, isFolder, path, virtual_path, parent=None):
        self.name = name
        if virtual_path is not None:
            self.virtual_path = f"{virtual_path}/{self.name}"
        else:
            self.virtual_path = f"/{self.name}"
        self.path = path
        self.is_Folder = isFolder
        if self.is_Folder is True:
            self.child = {}
        else:
            self.child = None
        if parent is not None:
            self.parent = parent
        elif name == "root":
            self.parent = None
        else:
            self.parent = "root"

    def get_name(self):
        return self.name

    def get_node(self):
        return self

    def get_child(self):
        return self.child

    def get_child_name(self):
        return list(self.child.keys())

    def get_virtual_path(self):
        return self.virtual_path

    def change_virtual_path(self, new_path):
        self.virtual_path = new_path

    def get_parent_path(self):
        return self.virtual_path.rstrip(f"/{self.get_name()}")

    def change_parent(self, new_parent):
        self.parent = new_parent

    def add_child(self, child):
        if child.get_name() not in self.get_child_name():
            self.child.update({child.get_name(), child})
            return True
        else:
            return False

    def have_child(self):
        if len(self.child) > 0:
            return True
        else:
            return False

    def remove_child(self, child):
        if child.get_name() in self.get_child_name():
            if child.have_child() is False:
                self.child.pop(child.get_name)
                return True
        else:
            return False

    def relate_remove(self, child):
        if child.get_name() in self.get_child_name():
            self.child.pop(child.get_name)


class FileTree:
    def __init__(self, save_path):
        save_tree = None
        if "FileTreeSaved.pickle" in os.listdir(save_path):
            save_tree = os.path.join(save_path, "FileTreeSaved.txt")
        if save_tree is None:
            self.root = TreeNode("root", True, None, None)
        else:
            with open('FileTreeSaved.pickle' 'rb') as tree_file:
                self.root = pickle.load(tree_file)

    def get_node(self, pos_list):
        pass

    def find_node(self, name, virtual_path):
        target = [[self.root]]
        next_generation = []
        while len(target) != 0:
            for part in target:
                for t in part:
                    if t.get_name() == name and t.get_virtual_path() == virtual_path:
                        return t.get_self()
                    next_generation.append(self.root.get_child())
            target = next_generation
            next_generation = []

        return None

    def move(self, former, latter):
        latter.change_parent(former.get_name())
        latter.change_virtual_path(f"{former.get_virtual_path()}/{latter.get_name()}")
        former.add_child(latter)

    def remove(self, node_name, node_path):
        x = self.find_node(node_name, node_path)
        if x is not None:
            parents = self.find_node(x.get_parent(), x.get_parent_path())
            if x.is_Folder is False:
                parents.remove_child(x)
            else:
                parents.relate_remove(x)

    def add(self, node, target_node):
        isr = target_node.add_child(node)
        if isr is True:
            node.change_parent(target_node.get_name())
            node.change_virtual_path(f"{target_node.get_virtual_path}/{node.get_name()}")




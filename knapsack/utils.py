from dataclasses import dataclass, field
import heapq
from typing import Any


@dataclass(order=True, frozen=True)
class PrioritizedItem:
    priority: int # taken here from the optimistic estimation (last index in the given tuple)
    item: Any=field(compare=False)
    

class PriorityQueue:
  """
  """
  def __init__(self):
    self.heap = []

  def push(self, element):
    """
    """
    heapq.heappush(self.heap, PrioritizedItem(priority=-element[-1], item=element))

  def pop(self):
    return heapq.heappop(self.heap).item

  def empty(self) -> bool:
    return len(self.heap) < 1
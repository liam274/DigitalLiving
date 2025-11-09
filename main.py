"""
群居 (Qunju)-數字生命模擬系統
一個模擬虛擬生命進化的人工生命項目

## 本項目視為最高機密

其實…這個項目也是在探索我們自己
"""
# © 2025 Liam Lei
# Licensed under CC BY-NC-ND 4.0
# See LICENSE file for details
import random
import math
import time
from dataclasses import dataclass,field
from typing import Generic,TypeVar,Optional,Any,Generator,TextIO,Callable,Union
from collections import deque,Counter
import string
import tkinter as tk
import os
import sys
import weakref
from array import array
import types
import heapq
from operator import attrgetter
import numpy as np
from names_dataset import NameDataset # type: ignore

# constants
POSITIVE_FEELINGS: tuple[str,...]=("happy","surprised","trusting","joyful",
                                      "anticipating","exciting","releaxed","love",
                                      "hopeful")
NEGATIVE_FEELINGS: tuple[str,...]=("sad","angry","disgusted","fearful","bored",
                                     "annoyed","stressed","nervous","guilty",
                                     "tiring")
POSITIVE_RANGE: int=len(POSITIVE_FEELINGS)
NEGATIVE_RANGE: int=len(NEGATIVE_FEELINGS)
FEELING_RANGE: int=POSITIVE_RANGE+NEGATIVE_RANGE
FEELINGS: tuple[str,...]=POSITIVE_FEELINGS+NEGATIVE_FEELINGS

POSTIVE_PERSONALITIES: tuple[str,...]=("positivity","attention_span","memory_index",
                "memory_range","calmness","curiosity","patience","memory_width",
                "hear_limit","shyness","far_think")
NEGATIVE_PERSONALITIES: tuple[str,...]=("negativity",)
PERSONALITIES: tuple[str,...]=POSTIVE_PERSONALITIES+NEGATIVE_PERSONALITIES

PRINTABLE: str=string.ascii_lowercase+string.ascii_uppercase+string.digits+" "

WEATHERS: tuple[str,...]=("sunny","cloudy","rainy","stormy","snowy","foggy")

FOODS: tuple[str,...]=("food","apple","banana")

# funcs

def split(l: list[Any],size: int)->list[list[Any]]:
    """split it"""
    return [l[i:i+size] for i in range(0,len(l),size)]
def most_frequent(l: list[Any],default: Any)->Any:
    """find out the most frequent"""
    if not l:
        return default
    data: Counter[Any]=Counter(l)
    max_freq: int=max(data.values())
    return random.choice(tuple(map(lambda kv:kv[0],
        filter(lambda kv:kv[1]==max_freq,data.items()))))
def random_string()->str:
    """return a random string"""
    return "".join(random.choice(PRINTABLE)
        for _ in range(random.randint(1,20)))
def shuffle(l: list[Any],repeat_weight: dict[Any,float]={})->list[Any]:
    """Shuffle the list, allowing some elements to repeat based on weight."""
    r: list[Any]=[None]*len(l)
    counter: Counter[Any]=Counter(l)
    last: Any=None
    basic: tuple[Any]=tuple(counter.keys())
    candidates: tuple[Any]
    for i in range(len(l)):
        basic=tuple(counter.keys())
        candidates=basic
        if last is not None and last in candidates:
            if random.random()<repeat_weight.get(last,.5):
                # bigger, easier to repeat
                candidates=tuple(filter(lambda c:c!=last,candidates)) or basic
        choice=random.choice(candidates)
        r[i]=choice
        counter[choice]-=1
        if counter[choice]<=0:
            del counter[choice]
        last=choice
    return r
def age2weight(age: float)->float:
    """convert age to weight"""
    v: float=0
    s: float=0
    if age<.333:
        v,s=29,5
    else:
        v,s=16,20
    return 80/(1+math.exp((v-age)/s))
def chose(l: list[Any],t: Any,count: int=1)->tuple[Any,...]:
    """chose without repeat"""
    # tuple(i for i in l if i is not t)
    return tuple(random.sample(tuple(filter(lambda i: i is not t, l)),count))
def mixture(a: dict[str,dict[str,float]],b: dict[str,dict[str,float]])\
    ->dict[str,dict[str,float]]:
    """mixture together, used in gene"""
    result: dict[str,dict[str,float]]={}
    temp: dict[str,float]={}
    u: float
    for name,dic in a.items():
        temp.clear()
        for nam,val in dic.items():
            u=random.random()
            temp[nam]=val*u+b[name][nam]*(1-u)
        result.update({name:temp})
    return result
def sort_chose(l: list[Any],t: Any,count: int=1)->tuple[Any,...]:
    """sort it and chose it"""
    # l=[i for i in l if i is not t]
    return tuple(heapq.nsmallest(count,tuple(filter(lambda i: i is not t,l)),
        key=lambda a:a.position_.square_distance(t.position_)))
def between(a: float,upper: float,lower: float):
    """check if is between"""
    return lower<a<upper
def format_duration(seconds: int)->str:
    """format the tick and output as date"""
    days: float
    remainder: float
    hours: float
    minutes: float
    days,remainder=divmod(seconds,86400)
    hours,remainder=divmod(remainder,3600)
    minutes,seconds=divmod(remainder,60)
    return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"
def personality_transform(data: dict[str,"personalities"],
    how_it_goes:dict[str,float])->dict[str,tuple[float,"personalities"]]:
    """transform personality"""
    return {key:(how_it_goes.get(key,.5*random.random()),per)for key,per in data.items()}
def clear_relationship(objId: int):
    """clear all relationship to the life_obj"""
    global voices
    life_obj: "life"=WORLD.lifes[objId]
    voices=dict(filter(lambda kv:kv[0][0] is not life_obj,voices.items()))
    for other in list(life_obj.friends):
        other.friends.discard(life_obj)
    for other in life_obj.hug2heat:
        other.hug2heat.discard(life_obj)
print_temp: list[str]=[]
def _print(*content: Any,sep: str=" ",end: str="\n",file: TextIO):
    """Print that will auto flush if it goes out of a range"""
    global print_temp
    print_temp.append(sep.join(str(i) for i in content)+end)
    if len(print_temp)>200:
        file.write("".join(print_temp))
        print_temp.clear()
# classes

## data
T=TypeVar("T")
class queue(Generic[T]):
    """A simple queue with max size"""
    __slots__=("data",)
    def __init__(self,max_size: int):
        self.data: deque[T]=deque(maxlen=max_size)
    def enqueue(self,item: T):
        """Add item in the deque"""
        self.data.append(item)
    def dequeue(self)->Optional[T]:
        """Remove the first given item"""
        return self.data.popleft() if self.data else None
    def top(self)->Optional[T]:
        """Get the earliest"""
        return self.data[0] if self.data else None
    def is_empty(self)->bool:
        """Check if the queue is empty"""
        return len(self.data)==0
    def __iter__(self)->Generator[T,Any,None]:
        """Iterator"""
        yield from self.data

## low-level classes
class position:
    """state 2d position"""
    __slots__=("data","name")
    def __init__(self,x: float,y: float,name: str):
        self.data=array("d",[x,y])
        self.name=name
    @property
    def x(self)->float:
        """return x"""
        return self.data[0]
    @x.setter
    def x(self,value: float):
        """set x"""
        self.data[0]=value
    @property
    def y(self)->float:
        """return y"""
        return self.data[1]
    @y.setter
    def y(self,value: float):
        """set y"""
        self.data[1]=value
    def square_distance(self,other: "position")->float:
        """return the euclidean distance without square root. Good for comparing"""
        return (self.x-other.x)**2+(self.y-other.y)**2
    def distance(self,other: "position")->float:
        """return euclidean distance"""
        return float(np.linalg.norm(np.array(self.data)-np.array(other.data)))
@dataclass(slots=True)
class emotion_stat:
    """state emotion"""
    name: str
    value: float
@dataclass(slots=True)
class personalities:
    """state personalities"""
    name: str
    value: float
@dataclass(slots=True)
class event:
    """state event"""
    name: str
    time: int
    venue: position
    personality: dict[str,tuple[float,personalities]]
    repeat_times: int=field(default_factory=int)
    # if the personalities are same, randomly change one
    feeling: dict[str,emotion_stat]=field(default_factory=dict[str,emotion_stat])
    what2do: list[tuple[Callable[...,Any],
        list[tuple[Callable[...,Any],tuple[Any,...]]]]]=field(
            default_factory=list[tuple[Callable[...,Any],
            list[tuple[Callable[...,Any],tuple[Any,...]]]]]
        )
    # list[
    #   tuple[
    #     Callable[...,Any], #function
    #     list[
    #       tuple[
    #         Callable[...,Any],
    #         tuple[Any,...] # arguments
    #       ]
    #     ]
    #   ]
    # ]
    relate_event: set["event"]=field(default_factory=set["event"])
    # So the change of the event feeling will also affect the related events.
    def change_feeling(self,n: float,specific: dict[str,float]={})->dict[str,emotion_stat]:
        """+ \\+ for positive feeling, - for negative feeling"""
        result: dict[str,emotion_stat]={
            i:emotion_stat(i,emo.value+(n*(-1)**(int(i
                in POSITIVE_FEELINGS)+1)))
            for i,emo in self.feeling.items()
        }
        for i,_ in result.copy().items():
            if i in specific:
                result[i].value+=specific[i]
        return result
    def __hash__(self)->int:
        return hash((self.name,self.time,self.venue))
    def end(self,positive: float)->dict[str,personalities]:
        """According to the result to modify the personality"""
        result: dict[str,personalities]={}
        positive*=.1
        for i in personality:
            personality[i].value*=positive
        return result
voices: dict[tuple[Union["environment","life"],position],str]={}
POINTLESS_EVENT: event=event("",0,position(0,0,""),{},0,{})

## environment classes

@dataclass(slots=True,frozen=True)
class biome:
    """define the biome, a data structure"""
    name: str
    temperature: float
    diural_tempature_difference: float
    water_required: float
    def __hash__(self)->int:
        """hash for dictonary"""
        return hash((self.name,self.temperature,
                     self.diural_tempature_difference,
                     self.water_required
                    ))
BIOMES: tuple[biome,...]=(
    biome("sea",20,-10,30),biome("river",19,-6,20),
    biome("grassland",20,-5,10),biome("snowland",0,-10,10),
    biome("snow-mountain",-10,-20,7),biome("highland",16,-5,5),
    biome("hill",22,-2,4),biome("cave",20,-10,3),
    biome("mountain",10,-12,2),biome("basin",30,-5,2),
    biome("desert",38,-42,0.1),biome("rocky",20,-5,0)
)
BIOME_SIZE: int=20
COLOR_MAP: dict[str,str]={
    "sea":"#0000FF","river":"#1E90FF",
    "grassland":"#7CFC00","snowland":"#FFFFFF",
    "snow-mountain":"#F0FFFF","highland":"#228B22",
    "hill":"#32CD32","cave":"#A9A9A9",
    "mountain":"#808080","basin":"#FFFF00",
    "desert":"#FFD700","rocky":"#D2B48C"
}

## mind-like classes
class memory:
    """memory"""
    __slots__=("data","recent_event")
    def __init__(self,data: dict[str,event],memory_width: int):
        self.data: dict[str,event]=data
        self.recent_event: queue[event]=queue(int(memory_width))
    def recall(self,name: str,memory_index: float)->event:
        """Recall memory and return. If the event is not recorded, it will fake one"""
        memory_index*=10
        if name in self.data:
            val: event=self.data[name]
            return event(name,val.time,val.venue,val.personality,
                val.time,
                {
                name:emotion_stat(name,
                    i.value+(i.value*random.uniform(.001,.2)/memory_index)) # noise factors
                 for name,i in val.feeling.items()
                })
        return POINTLESS_EVENT # We should fake some memory
    def forget(self,name: str)->Optional[event]:
        """Forget the curtain event"""
        return self.data.pop(name) # forget the first event remembered
    def remember_e(self,e: event):
        """remember an event"""
        if e.name in self.data:
            self.data[e.name].time+=1
            self.recent_event.enqueue(self.data[e.name])
            return
        self.recent_event.enqueue(e)
        self.data[e.name]=e
    def remember(self,name: str,t: int,venue: position,
            personality: dict[str,tuple[float,personalities]],
            feelings: dict[str,emotion_stat],float_index: float,
            what2do: list[tuple[Callable[...,Any],
            list[tuple[Callable[...,Any],tuple[Any,...]]]]]=[],
            related_event: set[event]=set()):
        """different from remember_e, this function builds the event obj for you"""
        e: event=event(name,t,venue,personality,1,
                       {i:emotion_stat(i,
                        feelings[i].value/(random.uniform(.001,.2)*float_index))
                         for i in FEELINGS} # noise factors
                         ,what2do,related_event
                       )
        if name in self.data:
            self.data[name].time+=1
            self.recent_event.enqueue(self.data[name])
            return
        self.data[name]=e
        self.recent_event.enqueue(e)
class mind:
    """consciousness"""
    __slots__=("thoughts","age","personality",
    "love","memory","concepts","feeling")
    def __init__(self,age: float,personality: dict[str,personalities]):
        self.thoughts: queue[dict[str,event]]=queue(int(personality["memory_range"].value))
        self.age: float=age
        self.personality: dict[str,personalities]=personality
        self.memory: memory=memory({},int(personality["memory_width"].value)) # type: ignore
        self.love: dict[str,event]={}
        self.concepts: dict[str,dict[str,event]]={"food":{"food":POINTLESS_EVENT}} # concepts
        self.feeling: dict[str,emotion_stat]={i:emotion_stat(i,0) for i in FEELINGS}
    def feel(self,name: str)->dict[str,emotion_stat]:
        """Feel a certain event"""
        # Should be more complicated
        return self.memory.data[name].feeling
    def forget(self,name: str)->Optional[event]:
        """simply forget it"""
        # should be more complicated
        return self.memory.forget(name) # just simply forget it
    def recall(self,name: str)->event:
        """recall the memory"""
        return self.memory.recall(name,self.personality["memory_index"].value) # remember the event
    def remember(self,name: str,time: int,venue: position,
                 personality: dict[str,tuple[float,personalities]],
                 feelings: dict[str,emotion_stat],float_index: float,
                 what2do: list[tuple[Callable[...,Any],
                list[tuple[Callable[...,Any],tuple[Any,...]]]]]=[],
                related_event: set[event]=set()):
        """remember some permanently(almost!)"""
        return self.memory.remember(name,time,venue,personality,
        feelings,float_index,what2do,related_event)
    def remember_e(self,e: event):
        """remember some event permanently(almost!)"""
        return self.memory.remember_e(e)
    def grow(self,time_delta: float):
        """your mind grows to be more stable and calm(unless the calmness is negative!)"""
        calmness: float=.1/self.personality["calmness"].value
        self.age+=time_delta/31536000
        for i in self.personality.values():
            # when you grow you become calmer
            i.value+=i.value*\
            random.uniform(-calmness,calmness)*time_delta/31536000
    def fake(self,e: event):
        """Fake something you don't wanna to"""
        feeling: dict[str,emotion_stat]=e.feeling
        remain: set[str]=set(FEELINGS)^set(e.feeling.keys())
        positivity: float=self.personality["positivity"].value
        negativity: float=self.personality["negativity"].value
        for i in remain:
            feeling[i]=emotion_stat(i,1-random.random()*
                        # simply fake it, with reversing the actual meaning
                        min(1,positivity if i in POSITIVE_FEELINGS else negativity)
                        )
        new: event=event(e.name,e.time,e.venue,
        personality_transform(self.personality,{"positivity":.5,
        "negativity":.5}),1,feeling)
        self.memory.remember_e(new)
        # This should be think here, which means we have to
        # implement the thinking more complicatedly
        # in this mind.
class unconscious_mind:
    """Unconscious mind"""
    __slots__=("personality","history_feeling_tick",
    "feeling","memory","thoughts","mind_","name")
    def __init__(self,name: str,personal: dict[str,personalities],mind_: mind):
        # Notice that the unconscious mind only have
        # the instincts of the beast
        self.personality: dict[str,personalities]=personal
        # initalize history feeling ticks.
        # This is using for patience of how frequent should a emotion explodes.
        self.history_feeling_tick: dict[str,int]={i:1 for i in FEELINGS}
        self.feeling: dict[str,emotion_stat]={i:emotion_stat(i,0) for i in FEELINGS}
        self.memory: memory=memory({},int(personality["memory_width"].value)) # type: ignore
        self.thoughts: queue[dict[str,event]]=queue(int(personality["memory_range"].value))
        self.mind_: mind=mind_
        self.name: str=name
    def think(self,thought: tuple[str,event])->tuple[bool,bool,dict[str,float]]:
        """Think if this should be in memory"""
        if not thought:
            return (False,False,{})
        want: bool=False # It stats if we should operate on this event.
        noticed: bool=False
        how_it_goes: dict[str,float]={}
        positivity: float=1/(1-self.personality["positivity"].value) # Optimize performance
        negativity: float=1/(1-self.personality["negativity"].value)
        patience: float=self.personality["patience"].value*.1
        attention_span: float=self.personality["attention_span"].value
        memory_index: float=self.personality["memory_index"].value
        value: float
        details,_event=thought
        _sum: float=0
        for name,emotion in _event.feeling.items():
            if self.history_feeling_tick[name]==0:
                continue
            value=emotion.value
            if name in POSITIVE_FEELINGS:
                self.feeling[name].value+=(value*positivity-self.feeling[name].value)*(
                    # more time no this emotion more this,same below
                    self.history_feeling_tick[name]/patience
                    # ten as the factor to low down personalities.patience
                    )
                _sum+=self.feeling[name].value
            else:
                self.feeling[name].value+=(value*negativity-self.feeling[name].value)\
                    *(self.history_feeling_tick[name]/patience)
                _sum-=self.feeling[name].value
        if abs(_sum)>attention_span:
            how_it_goes={"positivity":.6 if want else .4,
                        "negativity":.4 if want else .6,
                        "patience": 1/abs(_sum)
                        }
            # not matter it's too positive or too negative
            # it will still being remembered
            self.memory.remember(
                # flow the memory, the memory index is all about how good your memory is.
                details,_event.time,_event.venue,
                personality_transform(self.personality,how_it_goes),
                _event.change_feeling((-1)**(int(want)+1)*.4),
                memory_index
                )
            for i in _event.relate_event:
                self.affect_event(i,(-1)**(int(want)+1))
            temp: dict[str,event]=self.mind_.concepts.get(details,{})
            temp.update({details:_event})
            self.mind_.concepts[details]=temp
            noticed=True
            want=_sum>0
            for i in (POSITIVE_FEELINGS if want else NEGATIVE_FEELINGS):
                self.history_feeling_tick[i]=1
        else:
            # Actually we should also implement failure of thinking, too.
            pass
        _event.personality=personality_transform(self.personality,how_it_goes)
        # Implement changes of personality
        self.thoughts.enqueue({details:_event})
        return (want,noticed,how_it_goes)
    def most_want2do(self)->event:
        """return the most want to do event"""
        result: list[Union[str,float]]=["",sys.float_info.min]
        positivity: float=1/(1-self.personality["positivity"].value)
        negativity: float=1/(1-self.personality["negativity"].value)
        patience: float=self.personality["patience"].value*.1
        for name,_event in self.memory.data.items():
            _sum: float=0
            for nam,emo in _event.feeling.items():
                if nam in POSITIVE_FEELINGS:
                    _sum+=(emo.value*positivity-self.feeling[nam].value)\
                        *(self.history_feeling_tick[nam]/patience)
                else:
                    _sum-=(emo.value*negativity-self.feeling[nam].value)\
                        *(self.history_feeling_tick[nam]/patience)
            if abs(_sum)>abs(result[1]): # type: ignore
                result[1]=_sum
                result[0]=name
        if result[0]=="":
            return POINTLESS_EVENT
        for i in (POSITIVE_FEELINGS if result[1]>0 else NEGATIVE_FEELINGS): # type: ignore
            self.history_feeling_tick[i]=1
        return self.memory.data.get(result[0],POINTLESS_EVENT) # type: ignore
    def feel(self,name: str)->dict[str,emotion_stat]:
        """Feel a certain event"""
        # requires to be more complicated,
        # more instinctual
        return self.memory.data[name].feeling
    def forget(self,name: str)->Optional[event]:
        """simply forget it"""
        return self.memory.forget(name) # just simply forget it
    def recall(self,name: str)->event:
        """recall the memory"""
        return self.memory.recall(name,self.personality["memory_index"].value) # remember the event
    def remember(self,name: str,time: int,venue: position,
                 personality: dict[str,tuple[float,personalities]],
                 feelings: dict[str,emotion_stat],float_index: float,
                 what2do: list[tuple[Callable[...,Any],
                list[tuple[Callable[...,Any],tuple[Any,...]]]]]=[],
                related_event: set[event]=set()):
        """remember some permanently(almost!)"""
        return self.memory.remember(name,time,venue,personality,
            feelings,float_index,what2do,related_event)
    def remember_e(self,e: event):
        """remember some event permanently(almost!)"""
        return self.memory.remember_e(e)
    def affect_event(self,e: event,b: int=1,time: int=0):
        """affect the event in order to 飲水思源"""
        if time>self.personality["far_think"].value:
            # If think too long, then return
            _print(f"{self.name} woke up from thinking!",file=OUTPUT_FILE)
            return
        e.feeling=e.change_feeling(b*.4)
        for i in e.relate_event:
            self.affect_event(i,b,time+1)
class life:
    """ a living thing, more then a mind, or soul."""
    __slots__=("mind_","name","personality",
              "position_","love","tick","nutrition",
              "energy","is_alive","in_sleep","body_temp",
              "storage_fat","weight","extra_weight",
              "fat_index","water_content",
              "gene","current_biome","hug2heat","stomach",
              "stomach_max","unconscious","frequent",
              "freq_met","friends","max_age",
              "important_events","__weakref__")
    def __init__(self,name: str,
                 personality: dict[str,personalities],
                 pos: position):
        self.mind_: mind=mind(0,personality)
        self.name: str=name
        self.personality: dict[str,personalities]=personality
        self.position_: position=pos
        self.love: dict[str,event]={}
        self.tick: int=0
        self.energy: float=100.0
        # 單位血液內的血糖
        self.nutrition: float=100.0
        # 單位肌肉細胞&肝細胞內的糖原
        self.storage_fat: float=0.0
        # 脂肪細胞重量
        self.is_alive: bool=True
        self.in_sleep: bool=False
        self.weight: float=3.3 # in kg
        self.extra_weight: float=0
        self.body_temp: float=36.0 # in celsius
        self.fat_index: float=10
        self.water_content: float=85 # in percentage
        self.gene: dict[str,dict[str,float]]={"event_requirement":{}} # This forms the reaction
        self.current_biome: biome=self.get_current_biome()
        self.hug2heat: weakref.WeakSet[life]=weakref.WeakSet([self])
        self.stomach: float=0
        self.stomach_max: float=100
        self.unconscious: unconscious_mind=unconscious_mind(name,personality,self.mind_)
        self.frequent: dict[str,tuple[float,dict[str,event]]]={}
        self.freq_met: dict["life",int]={}
        self.friends: weakref.WeakSet[life]=weakref.WeakSet()
        self.max_age: float=1
        self.important_events: dict[str,set[event]]={}
        # of certain events. Omit teaching
    def get_age(self)->float:
        """return as year"""
        return self.tick/31536000
    def __hash__(self)->int:
        return hash(self.name)
    def unconscious_thinking(self,thought: tuple[str,event])->tuple[bool,bool,dict[str,float]]:
        """Think about something"""
        return self.unconscious.think(thought)
    def feel(self,name: str)->dict[str,emotion_stat]:
        """Feel about event"""
        return self.mind_.feel(name)
    def recall(self,name: str)->event:
        """recall the memory"""
        return self.mind_.recall(name)
    def forget(self,name: str)->Optional[event]:
        """forget the event"""
        return self.mind_.forget(name)
    def most_want2do(self)->event:
        """Instinctual"""
        return self.unconscious.most_want2do()
    def friend(self,obj: "life",how_it_goes: dict[str,float]):
        """see somebody as a friend"""
        temp: Callable[...,str]=lambda x: x # type: ignore
        e: event=event(
            f"friend {obj.name}",
            WORLD.time(),
            self.position_,
            personality_transform(self.personality,how_it_goes),1,
            self.mind_.feeling,
            [(self.friend,[(temp,(obj.name,))])] # type: ignore
        )
        self.mind_.memory.remember_e(e)
        self.friends.add(obj)
        self.communicate((random_string()+" "+obj.name,))
        temp: set[event]=self.important_events.get("friend",set())
        temp.add(e)
        self.important_events.update({"friend":temp})
    def request(self,obj: "life",request: str):
        """Request somebody for something"""
        obj.recieve(request)
    def recieve(self,request: str):
        """LOCI---Life Object Communcating Interface"""
        # Must recieve the request, then
        # do the specific action
        self.__dict__[request]()
    def grow(self,tick: int):
        """grow in body"""
        # simply add the self tick, which does not matters to the emotion, but body
        self.tick+=tick
        self.weight=age2weight(self.get_age())
        self.fat_index=self.storage_fat/self.weight
    def listen(self)->list[str]:
        """listen to the voices"""
        result: list[str]=[]
        hear_limit: float=self.personality["hear_limit"].value
        for (lif,position),string in voices.items():
            if abs(position.x-self.position_.x)>hear_limit\
                 or abs(position.y-self.position_.y)>hear_limit:
                continue
            if string in self.unconscious.memory.data:
                continue
            if lif is self:
                # skip own word
                continue
            s: str=""
            a: str
            for n in string:
                a=chr(
                    math.floor(ord(n)+
                               random.uniform(0,.5)*position.distance(self.position_)
                               )
                ) # Hard to hear
                if a in PRINTABLE:
                    s+=a
            result.append(s)
        return result
    def communicate(self,message: tuple[str]):
        """talk"""
        global voices
        for i in message:
            voices[(self,self.position_)]=i
            _print(self.name+":",i,file=OUTPUT_FILE)
    def percieve_event(self,e: event):
        """see an event and act la"""
        if self.unconscious_thinking((e.name,e))[1]:
            # remember based on noticed or not
            self.unconscious.remember_e(e)
    def is_dead(self)->bool:
        """check is dead"""
        if not self.is_alive:
            return True
        return False
    def get_current_biome(self)->biome:
        """get the certain biome"""
        # This expensive move will only be applyed on position changing.
        return WORLD.map[int(self.position_.y/BIOME_SIZE)][int(self.position_.x/BIOME_SIZE)]
    def nutrition2energy(self):
        """change nutrition to energy"""
        if self.energy<=40 and self.nutrition<0:
            if self.storage_fat>0 and self.nutrition<50:
                self.hydrolysis(10)
            self.energy+=min(10,self.nutrition)*1.1
            # 1克糖原能化為1.1克葡萄糖
            # 因為加入水解的水分子所以增加質量
            self.nutrition-=min(10,self.nutrition)
    def shiver(self):
        """shiver to gain heat"""
        shiver_heat: float=(36-self.body_temp)*.1
        self.body_temp+=shiver_heat
        self.energy-=shiver_heat*.5
    def sweat(self):
        """sweat to cooldown"""
        # According to Penman equation
        total_heat_load: float=.1*(self.current_biome.temperature-\
            self.body_temp)+1898.925*(self.body_temp-37)
        sweat_cooling: float=total_heat_load*min(.01,(self.body_temp-37)*.003)
        self.body_temp-=sweat_cooling
        self.water_content=(self.get_weight()*\
            self.water_content-sweat_cooling*.8)/self.get_weight()
        self.energy-=sweat_cooling*.01
    def get_stomach(self):
        """return self.stomach, for want2do attr in event"""
        return self.stomach
    def gift(self,dm: float,from_whom: "life"):
        """recieve gift, be thankful"""
        temp: event=POINTLESS_EVENT
        for i in self.important_events.get("friend",set()):
            if i.name==from_whom.name:
                temp=i
                break
        else:
            print(self.name+":","who sent me a gift that isn't my friend?")
            return
        self.stomach+=dm
        self.unconscious_thinking((f"a gift from {from_whom.name}",
            event(f"a gift from {from_whom.name}",WORLD.time(),
            self.position_,personality_transform(self.personality,{}),1,
            self.change_feeling(.5/self.personality["calmness"].value),
            [(self.gift,[(min,(10,self.get_stomach))])],{temp,})
            ))
        # This should add something, so the event will be related to
        # friendship, and improve the event(so do the relationship)
    def meet(self,another: tuple["life"]):
        """met somebody, chose to be friend or whatever"""
        result: tuple[bool,bool,dict[str,float]]
        for i in another:
            if i in self.friends:
                result=self.unconscious_thinking((f"share resource to {i.name}",
                    event(f"share resource to {i.name}",WORLD.time(),self.position_,
                    personality_transform(self.personality,{"positivity":.6}),1
                    ,self.change_feeling(.1))
                ))
                if all(result[:2]):
                    dm: float=max(5,self.stomach)
                    self.stomach-=dm
                    i.gift(dm,self)
            self.freq_met[i]=self.freq_met.get(i,0)+1
            result=self.unconscious_thinking((f"meet {i.name}",event(
                f"meet {i.name}",WORLD.time(),self.position_,
                personality_transform(self.personality,{}),1
                ,self.change_feeling(.1)
                )))
            if self.freq_met[i]>self.personality["attention_span"].value*6 and result[1]:
                self.friend(i,result[2])
    def move(self,_pos: Optional[position]=None,
             dx: Optional[float]=None,
             dy: Optional[float]=None)->Callable[...,tuple[float,float]]:
        """move to somewhere"""
        # This should consider the event in this position, and
        # change the idea.
        factor: float=self.personality["calmness"].value\
            *20*self.personality["shyness"].value
        if _pos:
            dx=_pos.x-self.position_.x
            dy=_pos.y-self.position_.y
        dx=dx or random.uniform(-self.energy,self.energy)/factor
        dy=dy or random.uniform(-self.energy,self.energy)/factor
        ox: float=self.position_.x
        oy: float=self.position_.y
        self.position_.x=max(0,min(WORLD.width-1,self.position_.x+dx))
        self.position_.y=max(0,min(WORLD.height-1,self.position_.y+dy))
        self.energy-=math.sqrt((ox-self.position_.x)**2+\
            (oy-self.position_.y)**2)*.00000001
        self.current_biome=self.get_current_biome()
        def pos()->tuple[float,float]:
            """The point of this function, is it's a callable.
            So, it can be stored, transported and used in want2do"""
            return (ox-self.position_.x,oy-self.position_.y)
        return pos
    def change_feeling(self,n: float,specific: dict[str,float]={})->dict[str,emotion_stat]:
        """+ \\+ for positive feeling, - for negative feeling"""
        result: dict[str,emotion_stat]={
            i:emotion_stat(i,emo.value+(n*(-1)**(int(i
                in POSITIVE_FEELINGS)+1))/self.personality["calmness"].value)
            for i,emo in self.unconscious.feeling.items()
        }
        for i,_ in result.copy().items():
            if i in specific:
                result[i].value+=specific[i]
        return result
    def dream(self):
        """Have a dream"""
        # should have a dream here
    def sleep(self):
        """sleep to recover energy, but lose nutrition"""
        self.energy=min(100,self.energy+0.001388888888888889)
        self.nutrition=max(0,self.nutrition-0.000462962962962963)
    def is_starving(self)->bool:
        """check if starving"""
        return self.nutrition<20
    def touch_food(self)->bool:
        """check if touching food"""
        for i in WORLD.obj:
            if abs(self.position_.x-i.pos.x)>5 or abs(self.position_.y-i.pos.y)>5:
                continue
            if self.position_.square_distance(i.pos)<=25:
                if i.name not in self.mind_.concepts.get("food",{}):
                    if random.random()>.1*(1+self.personality["curiosity"].value)/(self.stomach-80):
                        # don't wanna try
                        return False
                    _print(f"{self.name} tried new food!",file=OUTPUT_FILE)
                    self.unconscious_thinking(("try new food",
                        event("try new food",WORLD.time(),self.position_,
                              personality_transform(self.personality,{}),1
                              ,self.change_feeling(.1),[(self.touch_food,[])]
                        )
                    ))
                WORLD.obj.remove(i)
                return True
        return False
    def store_fat(self,limit: float=100):
        """store fat for energy"""
        if self.nutrition>limit:
            self.storage_fat+=self.nutrition-limit
            # dehydration condensation
            self.water_content+=(self.nutrition-limit)*.5
            self.extra_weight+=(self.nutrition-limit)*1.4
            # the density of fat is .9 g/ml, while water is 1 g/ml
            self.stomach-=(self.nutrition-limit)*.25
            self.nutrition-=self.nutrition-limit
            self.unconscious_thinking(("store fat",event(
                "store fat",WORLD.time(),self.position_,
                personality_transform(self.personality,{}),1,
                self.change_feeling(.5)
            )))
            self.fat_index=self.storage_fat/self.get_weight()
    def hydrolysis(self,amount: float)->bool:
        """do hydrolysis to fat."""
        if self.storage_fat<=0:
            return False
        self.nutrition+=min(self.storage_fat,amount)*2.57 # 2.34*1.1
        self.water_content-=min(self.storage_fat,amount)*.8
        self.fat_index=1-(amount/self.storage_fat)
        self.storage_fat-=amount
        return True
    def find_food(self):
        """find food to eat"""
        self.move()
        self.unconscious_thinking(("find food",event(
            "find food",WORLD.time(),self.position_,
            personality_transform(self.personality,{}),1,
            self.change_feeling(.5)
        )))
    def get_weight(self)->float:
        """return the exact weight"""
        return self.weight+self.extra_weight
    def hug2gain_heat(self,another: tuple["life",...],tiggered: bool=False):
        """hug to gain heat"""
        for i in another:
            self.hug2heat.add(i)
        self.unconscious_thinking(("hug to gain heat",event(
            "hug to gain heat",WORLD.time(),self.position_,
            personality_transform(self.personality,{"posivity":.6}),1,
            self.change_feeling(.7)
        )))
        # They should be more careful about hugging to gain heat
        # Because, if they hug together,
        # their movement will be bind to together, too.
        # Moreover, they can't see the thing back at them
        # And, if they once hugged, and STRONGLY want to move alone,
        # It can quit it by using request.
        # P.S. hug2gain_heat can improve relationship
        if tiggered:
            return
        for i in another:
            i.hug2gain_heat(tuple(set(another)^{i,}),True)
        _print(", ".join(map(attrgetter("name"),self.hug2heat)),"hug to gain heat!",file=OUTPUT_FILE)
    def temp_death(self,dt: float):
        """check for tempature, if its in some specific condition,
        it makes the object dies."""
        if self.body_temp>38:
            cooling_rate: float=.5+(1/((self.fat_index or 1)+.5))
            self.body_temp-=min((self.body_temp-36)*.1, cooling_rate)
            self.unconscious_thinking(("feel hot",event("feel hot",WORLD.time(),
                        self.position_,
                        personality_transform(self.personality,{}),1,
                        self.change_feeling(-.4))
                        ))
        if self.current_biome.temperature>38:
            if random.random()<.01:
                # hot adapt
                self.hydrolysis(.05)
        elif self.current_biome.temperature<25:
            if random.random()<.01:
                # cold adapt
                self.store_fat(95)
                self.nutrition2energy()
        if 34<self.body_temp<37:
            self.unconscious_thinking(("feel warm",event("feel warm",WORLD.time(),
                        self.position_,personality_transform(self.personality,{}),1
                        ,self.change_feeling(.6),
                        [(self.hug2gain_heat,[(sort_chose,(WORLD.lifes,self,2))])]
                    )))
        elif self.body_temp<34:
            self.unconscious_thinking(("feel cold",event("feel cold",WORLD.time(),
                        self.position_,personality_transform(self.personality,{}),1
                        ,self.change_feeling(-.4)
                    )))
        else:
            if self.fat_index>15 or dt>1:
                self.sweat()
            self.nutrition2energy()
            self.shiver()
        if self.body_temp>44:
            self.is_alive=False
        elif self.body_temp<30:
            self.is_alive=False
    def get_dead_reason(self)->str:
        """return dead reason in details"""
        r: set[str]={
            "heat" if self.body_temp>44 else "",
            "hypothermia" if self.body_temp<30 else "",
            "nature deat" if self.get_age()>=self.max_age else "",
            "dehydration" if self.water_content<25 else "",
            "starving" if self.energy<0 else "",
            "stomach rupture" if self.stomach>self.stomach_max*1.3 else ""
        }
        return ", ".join(filter(bool,r))
    def update(self):
        """update the mainloop"""
        if self.is_dead():
            return
        # grow
        self.mind_.grow(1)
        self.grow(1)
        # use up
        self.energy-=1.6534391534391535e-06
        self.water_content-=3.858024691358025e-06
        if self.nutrition<=40:
            self.hydrolysis(min(self.storage_fat,25))
        self.nutrition2energy()
        # death operations
        if self.get_age()>=self.max_age:
            self.is_alive=False
            return
        if self.water_content<25:
            self.is_alive=False
            return
        if self.energy<=0:
            self.is_alive=False
            return
        dt: float=(self.current_biome.temperature-self.body_temp)/\
            ((self.fat_index or 1)*536870912*len(self.hug2heat))
        # this can make them being stable
        self.body_temp+=dt
        self.temp_death(dt)
        if not self.in_sleep and self.energy<40:
            if self.storage_fat<80-self.nutrition:
                self.in_sleep=random.random()<.1
                if self.in_sleep:
                    _print(self.name,"is slept in starving!",file=OUTPUT_FILE)
            else:
                self.in_sleep=True
                _print(self.name,f"is in sleep at tick {WORLD.time()}!",file=OUTPUT_FILE)
        if self.in_sleep:
            self.sleep()
            self.dream()
            if between(abs(dt),3,2):
                if between(self.body_temp,41,39):
                    self.in_sleep=False
                    _print(self.name,f"woke up due to heat at tick {WORLD.time()}!",file=OUTPUT_FILE)
                elif between(self.body_temp,30,24):
                    self.in_sleep=False
                    _print(self.name,f"woke up due to cold at tick {WORLD.time()}!",file=OUTPUT_FILE)
            if self.energy>=80:
                self.in_sleep=False
                _print(self.name,f"woke up just at tick {WORLD.time()}!",file=OUTPUT_FILE)
            return
        if self.touch_food():
            _print(self.name,"found food!",file=OUTPUT_FILE)
            self.nutrition+=20
            self.water_content+=5
            self.store_fat()
            self.unconscious_thinking((
                "eat food",event("eat food",WORLD.time(),self.position_,
                        personality_transform(self.personality,{}),1
                        ,self.change_feeling(.5))
                        ))
            self.stomach+=5
        if between(self.stomach,self.stomach_max*.9,self.stomach_max*.7):
            self.unconscious_thinking((
                "be full",event("be full",WORLD.time(),
                        self.position_,personality_transform(self.personality,{}),1
                        ,self.change_feeling(.6)
                    )))
        elif self.stomach>self.stomach_max*.9:
            self.unconscious_thinking((
                "be too full",event("be too full",WORLD.time(),
                        self.position_,personality_transform(self.personality,{}),1
                        ,self.change_feeling(-.1)
                    )
            ))
        if self.stomach>self.stomach_max*1.3:
            self.is_alive=False
        """解放人性
        if self.storage_fat>0 and self.energy>90:
            a: life=chose(WORLD.lifes,self)[0]
            if a:
                self.sex(a)
            # """
        positivity: float=self.personality["positivity"].value
        calmness: float=self.personality["calmness"].value
        if not self.unconscious.memory.data:
            result: list[str]=self.listen()
            if not result:
                self.unconscious.feeling=self.change_feeling(-.7/positivity)
                _print(self.name+": I feel so lonely...",file=OUTPUT_FILE)
                return
            for i in result:
                # 聯想
                if i in self.mind_.concepts:
                    self.unconscious_thinking(list(self.mind_.concepts[i].items())[0])
                    break
            else:
                # 專注於某事
                choosen: str=random.choice(result)
                # later, I shell change it to sorting it of the simularity
                # from concepts, and find out the most positive one.
                self.unconscious.feeling=self.change_feeling(-.6/calmness)
                # Because of loness, feeling will drops
                self.unconscious_thinking((choosen,event(choosen,WORLD.time(),
                            self.position_,personality_transform(self.personality,{}),1
                            ,self.unconscious.feeling)
                        ))
            self.move() # move makes memory and feelings
            _print(self.name,"memory initialization...",file=OUTPUT_FILE)
        negativity: float=self.personality["negativity"].value
        if self.is_starving():
            self.unconscious_thinking(
                ("be starving",event("be starving",WORLD.time(),
                self.position_,personality_transform(self.personality,{}),1
                ,self.change_feeling(0,
                    {"stressed":negativity/(calmness*2)})))
            )
            self.unconscious_thinking(
                ("find food",event("find food",WORLD.time(),
                    self.position_,personality_transform(self.personality,{}),1
                    ,self.change_feeling(0,{"hopeful":.5/calmness})))
            )
        want: bool
        noticed: bool
        _: dict[str,float]
        for i in tuple(self.unconscious.thoughts)[::-1]:
            # review thoughts
            want,noticed,_=self.unconscious_thinking(tuple(i.items())[0])
            if noticed:
                _print(self.name+":","I"+(" don't"*int(not want))\
                    +" want to",tuple(i.keys())[0],file=OUTPUT_FILE)
        # defines the var type here, yet var1: type, var2: type is not supported
        name: str
        e: event
        name,e=next(reversed(self.unconscious.memory.data.items()))
        const1: float=1/positivity
        const2: float=-1/negativity
        if all(self.unconscious_thinking((name,e))[:2]):
            l: list[Union[float,dict[str,event]]]=\
                list(self.frequent.get(name,(0,self.mind_.concepts[name])))
            l[0]+=1 # type: ignore
            self.frequent[name]=tuple(l) # type: ignore
            del l
            for i in tuple(self.unconscious.thoughts.data[0].values())[0].what2do:
                # list[tuple[Callable[...,Any],list[tuple[Callable[...,Any],tuple[Any,...]]]]]
                i[0](*(func(*((n() if isinstance(n,types.FunctionType)
                    else n) for n in arg)) for func,arg in i[1]))
                _print(self.name,"is doing",tuple(self.unconscious.thoughts.data[0]\
                    .values())[0].name,file=OUTPUT_FILE)
            for t,(_name,emotion) in enumerate(e.feeling.items()):
                # loop through the feeling and change it if it's remembered
                self.unconscious.history_feeling_tick[_name]+=1
                value: float=emotion.value
                if t<POSITIVE_RANGE:
                    if value>=const1:
                        self.unconscious.history_feeling_tick[_name]=0
                    continue
                if value<=const2:
                    self.unconscious.history_feeling_tick[_name]=0
            if name not in self.mind_.concepts:
                # make concepts
                m: str=most_frequent(self.listen(),random_string())
                temp: dict[str,event]=self.mind_.concepts.get(m,{})
                temp.update({m:e})
                self.mind_.concepts.update({m:temp})
                self.communicate((m,))
                # If a word frequently appears, it should be recorded as concept
                # Likely but have to be updated.
        if self.water_content<50:
            self.unconscious_thinking(("be thirsty",event("be thirsty",
                        WORLD.time(),self.position_,
                        personality_transform(self.personality,{}),1
                        ,self.change_feeling(-.4)
                        ))
                    )
        # heat perform system
        self.unconscious_thinking((
            f"go to pos({self.position_.x},{self.position_.y})",
            event(f"go to pos({self.position_.x},{self.position_.y})",
                WORLD.time(),self.position_,
                personality_transform(self.personality,{}),1
                ,self.unconscious.feeling)
        ))
        most: event=self.most_want2do()
        dx: float
        dy: float
        c: Callable[...,tuple[float,float]]
        if most is not POINTLESS_EVENT:
            c=self.move(most.venue)
        else:
            c=self.move()
        dx,dy=c()
        dt=math.sqrt(dx*dx+dy*dy)*(self.fat_index or 1)/536870912
        self.body_temp+=dt
        self.temp_death(dt)
        result: list[str]=self.listen()
        for i in result:
            if i in self.mind_.concepts:
                self.mind_.concepts[i][i].time+=1
            elif self.unconscious.recall(i).time>\
                self.personality["attention_span"].value:
                # repeat too many times
                # I can't really understand this piece of code, just a guess.
                pass
        for i in self.unconscious.feeling:
            # calmdown
            self.unconscious.feeling[i].value-=1e-4
    def sex(self,another: "life")->Optional["life"]:
        """make a baby with another life"""
        if not isinstance(another,life): # type: ignore
            return None
        if self.position_.square_distance(another.position_)>10:
            return None # too far away
        personality: dict[str,personalities]={}
        for i in PERSONALITIES:
            # give random weights
            w1: float=random.random()
            personality[i]=personalities(
                i,
                (self.personality[i].value*w1+another.personality[i].value*(1-w1))+
                random.uniform(-0.01,0.01) # some mutation
            )
        pos: position=position(
            another.position_.x+random.uniform(-1,1),# some mutation
            another.position_.y+random.uniform(-1,1),# some mutation
            f"({self.name} & {another.name})'s baby birthplace"
        )
        baby: life=life(f"({self.name}&{another.name}-baby)",personality,pos)
        WORLD.append_life(baby)
        baby.gene=mixture(self.gene,another.gene)
        self.energy-=20
        another.energy-=20
        self.body_temp+=.5
        self.unconscious_thinking(("sex with "+another.name,event(
            "sex with "+another.name,WORLD.time(),self.position_,
            personality_transform(self.personality,{}),1
            ,self.change_feeling(.6,{"tiring":.4})
        )))
        another.unconscious_thinking(("sex with "+self.name,event(
            "sex with "+self.name,WORLD.time(),self.position_,
            personality_transform(self.personality,{}),1
            ,self.change_feeling(.6,{"tiring":.4})
        )))
        _print(self.name,"and",another.name,"made a baby:",baby.name,file=OUTPUT_FILE)
        return baby
class human(life):
    """human, subclass of life"""

## environment

@dataclass(slots=True)
class weather:
    """define the weather, a data structure"""
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    certain_biome: biome
    def update(self):
        """update the weather"""
        # just putting random numbers
        self.wind_speed=(self.certain_biome.temperature-self.temperature)/10
        self.temperature=(self.certain_biome.temperature+self.temperature)*self.wind_speed/2
        # find the biggest wind_speed delta neibour and change the direction.
        self.humidity-=self.temperature/100
@dataclass(slots=True)
class dobj:
    """define the data object, a data structure"""
    name: str
    pos: position
    def update(self):
        """update the object, usually useless. Stated here as a default, could be covered"""
class environment:
    """define the environment"""
    __slots__=("width","height","size","biomes",
                 "water_content","water","tick",
                 "obj","lifes","map","history")
    def __init__(self,width: int,height: int,
                 water_content: float,obj: list[dobj],lifes: list[life]):
        self.width: int=width*BIOME_SIZE
        self.height: int=height*BIOME_SIZE
        self.size: int=width*height
        self.biomes: list[biome]=[]
        self.water_content: float=water_content
        # water content*area*10 as the factor
        self.water: float=water_content*self.size*10
        self.tick: int=0
        self.obj: list[dobj]=obj
        self.lifes: list[life]=lifes
        self.map: list[list[biome]]=[]
        self.history: dict[str,int]={"food":0}
    def init(self):
        """initalize the environment"""
        self.generate_map()
    def time(self)->int:
        """return time in ticks"""
        return self.tick
    def generate_map(self):
        """generate the map in biomes"""
        map: list[biome]=[]
        i: biome=random.choice(BIOMES)
        while self.water>i.water_required and len(map)<self.size:
            map.append(i)
            self.water-=i.water_required
            i=random.choice(BIOMES)
        while len(map)<self.size:
            map.append(BIOMES[-1])
        self.map=split(shuffle(map,{BIOMES[0]:1}),self.width//BIOME_SIZE)
        for x,n in enumerate(self.map):
            for y,i in enumerate(n):
                for _ in range(int(max(i.water_required/10,1))):
                    if random.random()<\
                        self.water_content/10**int(math.log10(self.water_content)):
                        self.obj.append(dobj(random.choice(FOODS),
                                position(x*BIOME_SIZE+random.uniform(0,BIOME_SIZE),
                                    y*BIOME_SIZE+random.uniform(0,BIOME_SIZE),random.choice(FOODS)
                                )
                            )
                        )
        print(len(self.obj),file=sys.stderr)
        """
        root: tk.Tk=tk.Tk()
        root.title("Color Grid Map")
        frame: tk.Frame=tk.Frame(root)
        frame.pack(padx=10,pady=10)
        x_size: int=len(self.map)
        y_size: int=len(self.map[0])
        canvas: tk.Canvas=tk.Canvas(root,width=x_size*BIOME_SIZE+60,
            height=y_size*BIOME_SIZE+60,bg="white")
        canvas.pack(padx=10, pady=10)
        for col_idx in range(len(self.map[0])):
            canvas.create_text((col_idx+.5)*BIOME_SIZE+45,y_size*BIOME_SIZE+30,
                text=str(col_idx),font=("Arial",10,"bold"))
        for row_idx,row in enumerate(self.map):
            canvas.create_text(30,(y_size-row_idx-2)*BIOME_SIZE+45,
                text=str(row_idx),font=("Arial",10,"bold"))
            for col_idx,cell in enumerate(row):
                color: str=COLOR_MAP.get(cell.name,"#000000")
                x1: float=col_idx*BIOME_SIZE+45
                y1: float=row_idx*BIOME_SIZE+15
                x2: float=x1+BIOME_SIZE
                y2: float=y1+BIOME_SIZE
                canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline="black")
        root.mainloop()
        # """;tk.ACTIVE
    def append_life(self,*l: life):
        """add a life"""
        for i in l:
            self.lifes.append(i)
    def mainloop(self):
        """mainloop"""
        global voices
        # """
        for t,n in enumerate(self.map):
            for t2,m in enumerate(n):
                if random.random()<.05:
                    voices.update({
                        (self,position(t*BIOME_SIZE+random.uniform(0,BIOME_SIZE),
                            t2*BIOME_SIZE+random.uniform(0,BIOME_SIZE),"")):
                        random_string()
                    })
        if self.history["food"]==1024:
            for t,n in enumerate(self.map):
                for t2,m in enumerate(n):
                    for _ in range(max(m.water_required//10,1)): # type: ignore
                        if random.random()<\
                        self.water_content/10**int(math.log10(self.water_content)):
                            self.obj.append(dobj("food",
                                position(t*BIOME_SIZE+random.uniform(0,BIOME_SIZE),
                                    t2*BIOME_SIZE+random.uniform(0,BIOME_SIZE),"food")
                            ))
                    self.history["food"]=0
        self.history["food"]+=1
        # """
        self.tick+=1
        for n in self.obj:
            n.update()
        time: int=len(self.lifes)-1
        i: life
        while time>-1:
            i=self.lifes[time]
            i.update()
            if i.is_dead():
                _print(f"[{WORLD.time()}]",i.name,
                f"is dead due to ({i.get_dead_reason()}){" in dream" if i.in_sleep else ""
                }, in {i.current_biome.name}!",
                file=OUTPUT_FILE)
                _print(i.name,f"died at age {i.get_age():.2f} years old.",file=OUTPUT_FILE)
                _print(f"deleting {self.lifes[time].name}",file=OUTPUT_FILE)
                clear_relationship(time)
                self.lifes.pop(time)
            time-=1
            del i
        voices.clear()

# example usage
if __name__!="__main__":
    print("need to run as main lol",file=sys.stderr)
    sys.exit(0)
if os.path.exists("state.stat"):
    with open("state.stat","r",encoding="utf-8") as file:
        if file.readlines():
            print("quit due to another running process...",file=sys.stderr)
            sys.exit()
    with open("state.stat","w",encoding="utf-8") as file:
        file.write("on") # state machine is on
else:
    with open("state.stat","a+",encoding="utf-8") as file:
        file.write("on")
if os.path.exists("output.txt"):
    with open("output.txt","w",encoding="utf-8") as file:
        file.write("") # clear the file
OUTPUT_FILE: TextIO=open("output.txt","a",encoding="utf-8")
try:
    personality: dict[str,personalities]={
        "positivity":personalities("positivity",0.7),
        "negativity":personalities("negativity",0.3),
        "attention_span":personalities("attention_span",0.5),
        "memory_index":personalities("memory_index",1.0),
        "memory_range":personalities("memory_range",10),
        "calmness":personalities("calmness",0.4),
        "curiosity":personalities("curiosity",0.6),
        "patience":personalities("patience",0.4),
        "hear_limit":personalities("hear_limit",50),
        "shyness":personalities("shyness",.9),
        "memory_width":personalities("memory_width",6),
        "far_think":personalities("far_think",10)
    }
    adam_home: position=position(0,0,"adam's")
    WORLD: environment=environment(20,20,2/3,[],[])
    WORLD.init()
    adam=human("亞當",personality,adam_home)
    eve=human("夏娃",personality,position(5,5,"eve's"))
    init_popular: int=100
    nd: dict[str,list[str]]=NameDataset().get_top_names((init_popular-2)//2, # type: ignore
        use_first_names=True,country_alpha2="GB")["GB"]
    humans: list[human]=[adam,eve,*(human(i,personality,adam_home) for i in nd["M"]+nd["F"])]
    WORLD.append_life(*humans)
    FPS: int=6000
    interval: float=1/FPS
    # t: float=time.monotonic()
    start: float=time.time()
    duration: float=time.time()
    temp: float
    while 1:
        if not WORLD.lifes:
            _print(f"All lifes are dead in {WORLD.tick} ticks!",file=OUTPUT_FILE)
            print(f"All lifes are dead in {format_duration(WORLD.tick)}!",file=sys.stderr)
            break
        if WORLD.tick&16383==0 and WORLD.tick>0:
            print("16384 tick passed!",file=sys.stderr)
            temp=time.time()
            print(f"{16384/(temp-duration):.2f}FPS, consumed {(temp-duration):.2f} seconds")
            duration=temp
            OUTPUT_FILE.close()
            OUTPUT_FILE=open("output.txt","a+",encoding="utf-8") # type: ignore
            print("output file saved!")
        WORLD.mainloop()
        # time.sleep(interval-((time.monotonic()-t)%interval))
    else:
        print("???") # I don't know why I need this,
        # but it keeps the bug away
    delta: float=time.time()-start
    OUTPUT_FILE.close()
    print("Closed!",file=sys.stderr)
    print("Consumed",delta,"second.",file=sys.stderr)
    print(f"{WORLD.tick/delta:.2f} FPS",file=sys.stderr)
except KeyboardInterrupt:
    print("Quit due to Keyboard Interrupt...",file=sys.stderr)
finally:
    if print_temp:
        OUTPUT_FILE.write("".join(print_temp))
        print_temp.clear()
with open("state.stat","w",encoding="utf-8") as file:
    file.write("") # clear to state quit
"""
群居 (Qunju)-數字生命模擬系統
一個模擬虛擬生命進化的人工生命項目

## 本項目視為最高機密

其實…這個項目也是在探索我們自己
"""
import random
import math
import time
from dataclasses import dataclass,field
from typing import Generic,TypeVar,Optional,Any,Generator,TextIO, Callable
from collections import deque,Counter
import string
import numpy as np
import tkinter as tk
import os

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
                "memory_range","calmness","curiosity","patience","hear_limit")
NEGATIVE_PERSONALITIES: tuple[str,...]=("negativity",)
PERSONALITIES: tuple[str,...]=POSTIVE_PERSONALITIES+NEGATIVE_PERSONALITIES

PRINTABLE: str=string.ascii_lowercase+string.ascii_uppercase+string.digits+" "

WEATHERS: tuple[str,...]=("sunny","cloudy","rainy","stormy","snowy","foggy")

if os.path.exists("log.txt"):
    with open("log.txt","w",encoding="utf-8") as file:
        file.write("") # clear the file
LOGFILE: TextIO=open("log.txt","a",encoding="utf-8")

# funcs

def split(l: list[Any],size: int)->list[list[Any]]:
    """split it"""
    return [l[i:i+size] for i in range(0,len(l),size)]
def most_frequent(l: list[Any],default: Any)->Any:
    """find out the most frequent"""
    if not l:
        return default
    data: Counter[Any]=Counter(l)
    max_freq=max(data.values())
    return random.choice(tuple(k for k,v in data.items() if v==max_freq))
def random_string()->str:
    """return a random string"""
    return "".join(random.choice(PRINTABLE) for _ in range(random.randint(1,20)))
def shuffle(l: list[Any],repeat_weight: dict[Any,float]={}) ->list[Any]:
    """Shuffle the list,allowing some elements to repeat based on weight."""
    r: list[Any]=[]
    counter: Counter[Any]=Counter(l)
    last: Any=None
    for _ in range(len(l)):
        candidates=list(counter.keys())
        if last is not None and last in candidates:
            weight=repeat_weight.get(last,0.5)
            if random.random() < weight:
                # bigger, easier to repeat
                candidates=[c for c in candidates if c != last]
        if not candidates:
            candidates=list(counter.keys())
        choice=random.choice(candidates)
        r.append(choice)
        counter[choice] -= 1
        if counter[choice] <= 0:
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
def chose(l: list[Any],t: Any)->Any:
    r: Any=random.choice(l)
    while t is r:
        r=random.choice(l)
    return r
# classes

## data
T=TypeVar("T")
class queue(Generic[T]):
    """A simple queue with max size"""
    __slot__=("max_size","data")
    def __init__(self,max_size: int):
        self.max_size: int=max_size
        self.data: deque[T]=deque(maxlen=max_size)
    def enqueue(self,item: T):
        """Add item in the deque"""
        self.data.append(item)
    def dequeue(self) ->Optional[T]:
        """Remove the first given item"""
        return self.data.popleft() if self.data else None
    def top(self) ->Optional[T]:
        """Get the earliest"""
        return self.data[0] if self.data else None
    def is_empty(self):
        """Check if the queue is empty"""
        return len(self.data) == 0
    def __iter__(self)->Generator[T,Any,None]:
        """Iterator"""
        yield from self.data

## low-level classes
@dataclass(slots=True)
class position:
    """state 2d position"""
    x: float
    y: float
    name: str
    def distance(self,other: "position") ->float:
        """return the distance"""
        return float(np.linalg.norm(np.array([self.x,self.y])-np.array([other.x,other.y])))
    def __str__(self):
        """stringify"""
        return f"({self.x:.2f},{self.y:.2f})"
    def __eq__(self,other: Any):
        """check if equal"""
        return isinstance(other,position) and self.x == other.x and self.y == other.y
    def __hash__(self):
        """hash for dictonary"""
        return hash((self.x,self.y))
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
    feeling: dict[str,emotion_stat]=field(default_factory=dict[str,emotion_stat])
    what_to_do: list[str]=field(default_factory=list[str])
voices: dict[position,str]={}

## environment classes

@dataclass(slots=True)
class biome:
    """define the biome,a data structure"""
    name: str
    temperature: float
    diural_tempature_difference: float
    water_required: float
    def __hash__(self):
        """hash for dictonary"""
        return hash((self.name,self.temperature,
                     self.diural_tempature_difference,self.water_required
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
    __slot__=("data","recent_event")
    def __init__(self,data: dict[str,event]):
        self.data: dict[str,event]=data
        self.recent_event: queue[event]=queue(100)
    def recall(self,name: str,memory_index: float)->dict[str,emotion_stat]:
        """Recall memory and return. If the event is not recorded,it will fake one"""
        if name in self.data:
            return {
                name:emotion_stat(name,
                    i.value+(i.value*random.uniform(.001,.2)/(10*memory_index))) # noise factors
                 for name,i in self.data[name].feeling.items()
                }
        return {i:emotion_stat(i,(random.random() or .001)) for i in FEELINGS} # fake some memory
    def forget(self,name: str)->Optional[event]:
        """Forget the curtain event"""
        return self.data.pop(name,None) # forget the first event remembered
    def remember_e(self,e: event):
        """remember an event"""
        self.recent_event.enqueue(e)
        self.data[e.name]=e
    def remember(self,name: str,t: int,venue: position,
            feelings: dict[str,emotion_stat],float_index: float)->None:
        """different from remember_e,this function builds the event obj for you"""
        e: event=event(name,t,venue,
                       {i:emotion_stat(i,
                        feelings[i].value/(random.uniform(.001,.2)*float_index))
                         for i in FEELINGS} # noise factors
                       )
        self.data[name]=e
        self.recent_event.enqueue(e)
class mind:
    """mind"""
    __slots__=("thoughts","name","age","personality",
    "love","memory","history_feeling_tick","concepts",
    "feeling")
    def __init__(self,name: str,age: float,personality: dict[str,personalities]):
        self.thoughts: queue[dict[str,event]]=queue(int(personality["memory_range"].value))
        self.name: str=name
        self.age: float=age
        self.personality: dict[str,personalities]=personality
        self.memory: memory=memory({})
        self.love: dict[str,event]={}
        # initalize history feeling ticks.
        # This is using for patience of how frequent should a emotion explodes.
        self.history_feeling_tick: dict[str,int]={i:1 for i in FEELINGS}
        self.concepts: dict[str,dict[str,event]]={} # concepts
        self.feeling: dict[str,emotion_stat]={i:emotion_stat(i,0) for i in FEELINGS}
    def think(self,thought: dict[str,event])->tuple[bool,bool]:
        """Think if this should be in memory"""
        if not thought:
            return (False,False)
        want: bool=False # It stats if we should operate on this event.
        noticed: bool=False
        positivity: float=self.personality["positivity"].value # Optimize performance
        negativity: float=self.personality["negativity"].value
        patience: float=self.personality["patience"].value
        attention_span: float=self.personality["attention_span"].value
        memory_index: float=self.personality["memory_index"].value
        for details,_event in thought.items():
            _sum: float=0
            for name,emotion in _event.feeling.items():
                if self.history_feeling_tick[name]==0:
                    continue
                value: float=emotion.value
                if name in POSITIVE_FEELINGS:
                    self.feeling[name].value=(value-self.feeling[name].value)*positivity*( # more time no this emotion more this,same below
                        (self.history_feeling_tick[name]*10)/patience
                        # ten as the factor to low down personalities.patience
                        )
                    _sum+=self.feeling[name].value
                else:
                    self.feeling[name].value+=(value-self.feeling[name].value)*negativity*((self.history_feeling_tick[name]*10)/patience)
                    _sum-=self.feeling[name].value
            if abs(_sum)>attention_span:
                # not matter it's too positive or too negative
                # it will still being remembered
                self.memory.remember(
                    # flow the memory,the memory index is all about how good your memory is.
                    details,_event.time,_event.venue,_event.feeling,memory_index
                    )
                temp: dict[str,event]=self.concepts.get(details,{})
                temp.update({details:_event})
                self.concepts[details]=temp
                noticed=True
                want=_sum>0
                if want:
                    for i in POSITIVE_FEELINGS:
                        self.history_feeling_tick[i]=1
                else:
                    for i in NEGATIVE_FEELINGS:
                        self.history_feeling_tick[i]=1
        self.thoughts.enqueue(thought)
        return (want,noticed)
    def feel(self,name: str)->dict[str,emotion_stat]:
        """Feel a certain event"""
        return self.memory.data[name].feeling
    def forget(self,name: str)->Optional[event]:
        """simply forget it"""
        return self.memory.forget(name) # just simply forget it
    def recall(self,name: str)->dict[str,emotion_stat]:
        """recall the memory"""
        return self.memory.recall(name,self.personality["memory_index"].value) # remember the event
    def remember(self,name: str,time: int,venue: position,
                 feelings: dict[str,emotion_stat],float_index: float):
        """remember some permanently(almost!)"""
        return self.memory.remember(name,time,venue,feelings,float_index)
    def remember_e(self,e: event):
        """remember some event permanently(almost!)"""
        return self.memory.remember_e(e)
    def grow(self,time_delta: float):
        """your mind grows to be more stable and calm(unless the calmness is negative!)"""
        calmness: float=self.personality["calmness"].value
        self.age+=time_delta/31536000
        for i in self.personality.values():
            # when you grow you become calmer
            i.value+=i.value* \
            random.uniform(-0.1/calmness,0.1/calmness)*time_delta/31536000
    def fake(self,e: event):
        """Fake something you don"t wanna to"""
        feeling: dict[str,emotion_stat]=e.feeling
        remain: list[str]=list(FEELINGS)
        positivity: float=self.personality["positivity"].value
        negativity: float=self.personality["negativity"].value
        for i in e.feeling:
            if i in remain:
                del remain[remain.index(i)] # left the one we don"t have
        for i in remain:
            feeling[i]=emotion_stat(i,1-random.random()*
                        # simply fake it,with reversing the actual meaning
                        min(1,positivity if i in POSITIVE_FEELINGS else negativity)
                        )
        new: event=event(e.name,e.time,e.venue,feeling)
        self.memory.remember_e(new)
class life:
    """ a living thing,more then a mind,or soul."""
    __slot__=("mind","name","age","personality",
              "position","feeling","love","tick",
              "energy","is_alive","in_sleep","nutrition",
              "storage_fat","weight","extra_weight","body_temp",
              "fat_index","dead_reason","water_content",
              "gene")
    def __init__(self,name: str,age: float,
                 personality: dict[str,personalities],pos: position):
        self.mind: mind=mind(name,age,personality)
        self.name: str=name
        self.age: float=age
        self.personality: dict[str,personalities]=personality
        self.position: position=pos
        self.love: dict[str,event]={}
        self.tick: int=0
        self.energy: float=100.0
        self.is_alive: bool=True
        self.in_sleep: bool=False
        self.nutrition: float=100.0
        self.storage_fat: float=0.0
        self.weight: float=2.0 # in kg
        self.extra_weight: float=0
        self.body_temp: float=36.0 # in celsius
        self.fat_index: float=10
        self.dead_reason: str=""
        self.water_content: float=85 # in percentage
        self.gene: dict[str,float] # This forms the reaction
        # of certain events. Omit teaching
    def think(self,thought: dict[str,event])->tuple[bool,bool]:
        """Think about something"""
        return self.mind.think(thought)
    def feel(self,name: str)->dict[str,emotion_stat]:
        """Feel about event"""
        return self.mind.feel(name)
    def recall(self,name: str)->dict[str,emotion_stat]:
        """recall the memory"""
        return self.mind.recall(name)
    def forget(self,name: str)->Optional[event]:
        """forget the event"""
        return self.mind.forget(name)
    def friend(self,obj: mind):
        """see somebody as a friend"""
        self.mind.memory.remember(
            obj.name,
            WORLD.time(),
            self.position,
            self.mind.feeling,
            self.personality["calmness"].value
            ) # memorize the new friend
    def grow(self,tick: int):
        """grow in body"""
        # simply add the self tick,which does not matters to the emotion,but body
        self.tick+=tick
        self.age+=tick/31536000
        self.weight=age2weight(self.age)
        self.storage_fat=self.weight*self.fat_index
    def listen(self)->list[str]:
        """listen to the voices"""
        global voices
        result: list[str]=[]
        for position,string in voices.items():
            if string in self.mind.memory.data:
                continue
            s: str=""
            for n in string:
                a: str=chr(
                    math.floor(ord(n)+
                               random.uniform(0,.5)*position.distance(self.position)
                               )
                ) # Hard to hear
                if a in PRINTABLE or \
                    position.distance(self.position)>self.personality["hear_limit"].value:
                    s+=a
            result.append(s)
        return result
    def communicate(self,message: list[str]):
        """talk"""
        global voices
        for i in message:
            voices[self.position]=i
            print(self.name+":",i,file=LOGFILE)
    def percieve_event(self,e: event)->None:
        """see an event and act la"""
        if self.think({e.name:e})[1]:
            self.mind.remember_e(e)
    def is_dead(self)->bool:
        """check is dead"""
        if not self.is_alive:
            self._del__()
            return True
        return False
    def _del__(self):
        """run when deleted(dead)"""
        print(f"[{WORLD.time()}]",self.name,f"is dead due to {self.dead_reason}, in biome {self.get_certain_biome().name}!",file=LOGFILE)
    def get_certain_biome(self)->biome:
        """get the certain biome"""
        return WORLD.map[int(self.position.y/BIOME_SIZE)][int(self.position.x/BIOME_SIZE)]
    def nutrition2energy(self):
        """change nutrition to energy"""
        if self.energy<=40 and self.nutrition<0:
            if self.storage_fat>0 and self.nutrition<50:
                self.nutrition+=min(10,self.storage_fat)
                self.storage_fat-=min(10,self.storage_fat)
            self.energy+=min(10,self.nutrition)
            self.nutrition-=min(10,self.nutrition)
    def shiver(self):
        """shiver to gain heat"""
        shiver_heat: float=(36-self.body_temp)*.1
        self.body_temp+=shiver_heat
        self.energy-=shiver_heat*.5
    def sweat(self):
        """sweat to cooldown"""
        sweat_cooling: float=(self.body_temp-37.5)*0.2
        self.body_temp-=sweat_cooling
        self.water_content-=sweat_cooling/2
    def move(self,dx: Optional[float]=None,dy: Optional[float]=None)->Callable[...,Any]:
        """move to somewhere"""
        # This should consider the event in this position, and 
        # change the idea.
        calmness: float=self.personality["calmness"].value
        dx=dx or random.uniform(-self.energy,self.energy)/calmness
        dy=dy or random.uniform(-self.energy,self.energy)/calmness
        ox: float=self.position.x
        oy: float=self.position.y
        self.position.x=max(0,min(WORLD.width-1,self.position.x+dx))
        self.position.y=max(0,min(WORLD.height-1,self.position.y+dy))
        self.energy-=math.sqrt((ox-self.position.x)**2+(oy-self.position.y)**2)*.1
        def pos()->tuple[float,float]:
            return (ox-self.position.x,oy-self.position.y)
        return pos
    def change_feeling(self,n: float,specific: dict[str,float]={})->dict[str,emotion_stat]:
        """+ + for positive feeling, - for negative feeling"""
        result: dict[str,emotion_stat]={
            i:emotion_stat(i,emo.value+(n*(-1)**(int(i in POSITIVE_FEELINGS)+1))/self.personality["calmness"].value)
            for i,emo in self.mind.feeling.items()
        }
        for i,_ in result.copy().items():
            if i in specific:
                result[i].value+=specific[i]
        return result
    def dream(self):
        pass
    def update(self):
        """update the mainloop"""
        # grow
        self.mind.grow(1)
        self.grow(1)
        # use up
        self.nutrition-=.01
        self.energy-=.01
        self.water_content-=.01
        self.nutrition2energy()
        # death operations
        if self.water_content<25:
            self.is_alive=False
            self.dead_reason="dehydration"
            return
        if self.energy<=0:
            if self.nutrition<=0:
                self.hydrolysis(min(self.storage_fat,100))
            else:
                self.nutrition2energy()
            self.is_alive=False
            self.dead_reason="starving"
            return
        if not self.in_sleep and self.energy<20:
            if self.storage_fat<80-self.nutrition:
                self.in_sleep=random.random()<.1
                print(self.name,f"is slept in starving!",file=LOGFILE)
            else:
                self.in_sleep=True
            if self.in_sleep:
                print(self.name,f"is in sleep at tick {WORLD.time()}!",file=LOGFILE)
        if self.in_sleep:
            self.sleep()
            self.dream()
            if self.energy>=80:
                self.in_sleep=False
                print(self.name,f"woke up just at tick {WORLD.time()}!",file=LOGFILE)
            else:
                self.hydrolysis(80-self.nutrition)
            return
        if self.touch_food():
            print(self.name,"found food!",file=LOGFILE)
            self.nutrition+=20
            self.water_content+=3
            self.store_fat()
            self.think({"eat food":event("eat food",WORLD.time(),self.position,
                        self.change_feeling(.5))
                        })
        """
        if self.storage_fat>0 and self.energy>90:
            self.sex(chose(WORLD.lifes,self))
            # """
        if not self.mind.memory.data:
            result: list[str]=self.listen()
            if not result:
                self.change_feeling(-.7/self.personality["positivity"].value)
                return
            for i in result:
                if i in self.mind.concepts:
                    self.think(self.mind.concepts[i])
                    break
            else:
                choosen: str=random.choice(result)
                self.think({choosen:event(choosen,WORLD.time(),
                            self.position,self.mind.feeling)
                        })
                # later, I shell change it to sorting it of the simularity from concepts, and find out the most positive one
            return
        positivity: float=self.personality["positivity"].value
        negativity: float=self.personality["negativity"].value
        calmness: float=self.personality["calmness"].value
        if self.is_starving():
            self.mind.feeling["stressed"].value+=negativity/(calmness*2)
            self.think({
                "be starving":event("be starving",WORLD.time(),self.position,self.mind.feeling),
                "find food":event("find food",WORLD.time(),self.position,self.change_feeling(0,{"hopeful":.5/calmness}))
                }
            )
        for i in tuple(self.mind.thoughts):
            want: bool
            noticed: bool
            want,noticed=self.think(i)
            if noticed:
                print(self.name+":","I"+" don't"*int(not want)+" want to",tuple(i.keys())[0],file=LOGFILE)
        # defines the var type here,yet var1: type,var2: type is not supported
        name: str
        e: event
        name,e=next(reversed(self.mind.memory.data.items()))
        const1: float=1/positivity
        const2: float=-1/negativity
        if self.think({name:e})[1]:
            for i in tuple(self.mind.thoughts.data[0].values())[0].what_to_do:
                self.__dict__[i]()
            for t,(_name,emotion) in enumerate(e.feeling.items()):
                # loop through the feeling and change it if it's remembered
                self.mind.history_feeling_tick[_name]+=1
                value: float=emotion.value
                if t<POSITIVE_RANGE:
                    if value>=const1:
                        self.mind.history_feeling_tick[_name]=0
                    continue
                if value<=const2:
                    self.mind.history_feeling_tick[_name]=0
            if name not in self.mind.concepts: # make concepts
                m: str=most_frequent(self.listen(),random_string())
                self.mind.concepts.update({m:{m:e}})
                self.communicate([m])
        if self.water_content<60:
            self.think({"be thirsty":event("be thirsty",WORLD.time(),self.position,
                        self.change_feeling(-.4)
                        )}
                    )
        # heat perform system
        dx: float
        dy: float
        c: Callable[...,tuple[float,float]]=self.move()
        dx,dy=c()
        dt: float=math.sqrt(dx*dx+dy*dy)*(self.fat_index or 1)/(400)+\
            (self.get_certain_biome().temperature-self.body_temp)/((self.fat_index or 1)*20)
        self.body_temp+=dt
        if self.body_temp>38:
            cooling_rate: float=.5+(1/((self.fat_index or 1)+.5))
            self.body_temp-=min((self.body_temp-36)*.1, cooling_rate)
            self.think({"be hot":event("be hot",WORLD.time(),
                        self.position,self.change_feeling(-.4))
                        })
        if self.get_certain_biome().temperature>38:
            if random.random()<.01:
                # hot adapt
                self.hydrolysis(.05)
        elif self.get_certain_biome().temperature<25:
            if random.random()<.01:
                # cold adapt
                self.store_fat(90)
                self.nutrition2energy()
        if self.body_temp>38:
            if self.fat_index>15 or dt>1:
                self.sweat()
        elif self.body_temp<30:
            self.think({"be cold":event("be cold",WORLD.time(),
                        self.position,self.change_feeling(-.2)
                    )})
            self.nutrition2energy()
            self.shiver()
        elif 34<self.body_temp<37:
            self.think({"be warm":event("be warm",WORLD.time(),
                        self.position,self.change_feeling(.2)
                    )})
        if self.body_temp>44:
            self.is_alive=False
            self.dead_reason="heat"
        elif self.body_temp<30:
            self.is_alive=False
            self.dead_reason="hypothermia"
    def sex(self,another: "life")->Optional["life"]:
        """make a baby with another life"""
        if not isinstance(another,life): # type: ignore
            return None
        if self.position.distance(another.position)>10:
            return None # too far away
        personality: dict[str,personalities]={}
        for i in PERSONALITIES:
            # give random weights
            w1: float=random.random()
            w2: float=random.random()
            personality[i]=personalities(
                i,
                (self.personality[i].value*w1+another.personality[i].value*w2)/(w1+w2)+
                random.uniform(-0.1,0.1) # some mutation
            )
        pos: position=position(
            another.position.x+random.uniform(-1,1),# some mutation
            another.position.y+random.uniform(-1,1),# some mutation
            f"{self.name} & {another.name}'s baby brithplace"
        )
        baby: life=life(f"{self.name}&{another.name}-baby",0,personality,pos)
        WORLD.append_life(baby)
        self.energy-=20
        another.energy-=20
        self.body_temp+=.5
        self.think({"sex with "+another.name:event(
            "sex with "+another.name,WORLD.time(),self.position,
            self.change_feeling(.6,{"tiring":.4})
        )})
        another.think({"sex with "+self.name:event(
            "sex with "+self.name,WORLD.time(),self.position,
            self.change_feeling(.6,{"tiring":.4})
        )})
        print(self.name,"and",another.name,"made a baby:",baby.name,file=LOGFILE)
        return baby
    def sleep(self):
        """sleep to recover energy, but lose nutrition"""
        self.energy=min(100,self.energy+.005)
        self.nutrition=max(0,self.nutrition-.002)
    def is_starving(self)->bool:
        """check if starving"""
        return self.nutrition<20
    def touch_food(self)->bool:
        """check if touching food"""
        for i in WORLD.obj:
            if abs(self.position.x-i.pos.x)>5 or abs(self.position.y-i.pos.y)>5:
                continue
            if self.position.distance(i.pos)<5 and "food"==i.name:
                return True
        return False
    def store_fat(self,limit: float=100):
        """store fat for energy"""
        if self.nutrition>limit:
            self.storage_fat+=self.nutrition-limit
            # dehydration condensation
            self.water_content+=(self.nutrition-limit)*.5
            self.extra_weight+=(self.nutrition-limit)*.9
            # the density of fat is .9g/ml
            self.nutrition-=self.nutrition-limit
            self.think({"store fat":event(
                "store fat",WORLD.time(),self.position,
                self.change_feeling(.5)
            )})
            self.fat_index=self.storage_fat/self.get_weight()
    def hydrolysis(self,count: float)->bool:
        """do hydrolysis to fat"""
        if self.storage_fat<=0:
            return False
        self.nutrition+=self.storage_fat*count
        self.water_content-=self.storage_fat*count*.8
        self.fat_index*=1-count
        self.storage_fat*=1-count
        return True
    def find_food(self):
        """find food to eat"""
        self.move()
        self.think({"find food":event(
            "find food",WORLD.time(),self.position,
            self.change_feeling(.5)
        )})
    def get_weight(self)->float:
        """return the exact weight"""
        return self.weight+self.extra_weight
class human(life):
    pass

## environment
@dataclass(slots=True)
class weather:
    """define the weather,a data structure"""
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    def update(self):
        pass
@dataclass(slots=True)
class dobj:
    """define the data object,a data structure"""
    name: str
    pos: position
    def update(self):
        pass
class environment:
    """define the environment"""
    __slots__=("width","height","size","biomes",
                 "water_content","water","tick","obj","lifes","map")
    def __init__(self,width: int,height: int,
                 water_content: float,obj: list[dobj],lifes: list[life]):
        self.width: int=width*BIOME_SIZE
        self.height: int=height*BIOME_SIZE
        self.size: int=width*height
        self.biomes: list[biome]=[]
        self.water_content: float=water_content
        self.water: float=water_content*self.size*10 # water content*area*10 as the factor
        self.tick: int=0
        self.obj: list[dobj]=obj
        self.lifes: list[life]=lifes
        self.map: list[list[biome]]=[]
    def init(self):
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
                    if random.random()<self.water_content/10**math.log10(self.water_content):
                        self.obj.append(dobj("food",
                                        position(x+random.uniform(0,BIOME_SIZE),y+random.uniform(0,BIOME_SIZE),"food")
                                        )
                        )
        print(len(self.obj))
        """
        root: tk.Tk=tk.Tk()
        root.title("Color Grid Map")
        frame: tk.Frame=tk.Frame(root)
        frame.pack(padx=10,pady=10)
        x_size: int=len(self.map)
        y_size: int=len(self.map[0])
        canvas: tk.Canvas=tk.Canvas(root, width=x_size*BIOME_SIZE+60, height=y_size*BIOME_SIZE+60, bg="white")
        canvas.pack(padx=10, pady=10)
        for col_idx in range(len(self.map[0])):
            canvas.create_text((col_idx+.5)*BIOME_SIZE+45, y_size*BIOME_SIZE+30, text=str(col_idx), font=("Arial", 10, "bold"))
        for row_idx,row in enumerate(self.map):
            canvas.create_text(30, (y_size-row_idx-2)*BIOME_SIZE+45, text=str(row_idx), font=("Arial", 10, "bold"))
            for col_idx,cell in enumerate(row):
                color: str=COLOR_MAP.get(cell.name,"#000000")
                x1: float=col_idx*BIOME_SIZE+45
                y1: float=row_idx*BIOME_SIZE+15
                x2: float=x1+BIOME_SIZE
                y2: float=y1+BIOME_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        root.mainloop()
        # """;tk.ACTIVE
    def append_life(self,l: life):
        """add a life"""
        self.lifes.append(l)
    def mainloop(self):
        """mainloop"""
        global voices
        self.tick+=1
        for i in self.obj:
            i.update()
        deled: int=0
        for time,i in enumerate(self.lifes[:]):
            i.update()
            if i.is_dead():
                print(i.name,f"died at age {i.age:.2f} years old.",file=LOGFILE)
                del self.lifes[time-deled]
                deled+=1
        voices.clear()

# example usage
personality={
    "positivity":personalities("positivity",0.7),
    "negativity":personalities("negativity",0.3),
    "attention_span":personalities("attention_span",0.5),
    "memory_index":personalities("memory_index",1.0),
    "memory_range":personalities("memory_range",10),
    "calmness":personalities("calmness",0.4),
    "curiosity":personalities("curiosity",0.6),
    "patience":personalities("patience",0.4),
    "hear_limit":personalities("hear_limit",50)
}
adam_home: position=position(0,0,"adam's")
adam=human("亞當",0,personality,adam_home)
eve=human("夏娃",0,personality,position(5,5,"eve's"))
# Generate a very positive feeling dict
WORLD: environment=environment(20,20,2/3,[],[adam,eve])
# adam.sex(eve)
WORLD.init()
FPS: int=6000
interval: float=1/FPS
t: float=time.monotonic()
while 1:
    if not WORLD.lifes:
        print(f"All lifes are dead in {WORLD.tick} ticks!",file=LOGFILE)
        break
    if WORLD.tick&4095==0 and WORLD.tick>0:
        # LOGFILE.close()
        LOGFILE=open("log.txt","a+",encoding="utf-8") # type: ignore
        print("4096 tick passed!")
    WORLD.mainloop()
    time.sleep(interval-((time.monotonic()-t)%interval))
else:
    print("???")
LOGFILE.close()
print("Closed!")
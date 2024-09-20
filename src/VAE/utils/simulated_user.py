from src.VAE.utils.imports import *

class SimUser:
    def __init__(self, rewards : list[float], labels : list[str]) -> None:
        self.rewards = rewards
        self.labels = labels

        self.index_to_terrain = {
            i : labels[i] for i in range(len(self.rewards))
        }

#        paired = list(zip(self.rewards, self.labels))
#        paired_sorted = sorted(paired, key=lambda x: x[0], reverse=False)

        self.terrain_order = self.get_preferential_order({l : r for l,r in zip(self.labels, self.rewards)})
        #[category for _, category in paired_sorted]

        print(f"TERRAIN ORDER: {self.terrain_order}")
    
    def get_preferential_order(self, reward_dict : dict) -> list[str]:
        """
        Determines a list of terrains in preferential order (i.e. sorts based
        on their reward values.)

        Args:
            reward_dict (dict): Reward dictionary mapping terrain types (str) to 
                                weight values (floats).

        Returns:
            list[str]: Returns a list of terrain types sorted based on the rewards.
        """
        paired_rewards = list(reward_dict.items())
        sorted_rewards = sorted(paired_rewards, key=lambda x : x[1])
#        print(f"SORTED REWARDS: {sorted_rewards}")
        return [c for c, _ in sorted_rewards]

    
    def check_distribution(self, other_rewards : list[str]) -> int:
        """
        Determines if the given distribution of reward weights has the same 
        preferential ordering as the simulated user. If not, reports how
        many of those weights were out of order. Used to make a trajectory
        selection in preference learning. 

        Args:
            other_rewards (list[str]): List of terrain types in preferential order. 
        Returns: 
            alignment_count (int): How many terrains were out of order. 
        """

        assert len(other_rewards) == len(self.rewards)

        print("Other Rewards:::")
        print(other_rewards)
        print("Ground Truth:::")
        print(self.terrain_order)
        print(self.rewards)

        alignment_count = 0
        for i, terrain in enumerate(self.terrain_order):
            if other_rewards[i] != terrain:
                alignment_count += 1

     #   alignment_count = 0
     #   for i in range(len(self.terrain_order)):
     #       if other_rewards[i] != self.terrain_order[i]:
     #           alignment_count += 1
        
#        print(f"ALIGNMENT COUNT: {alignment_count}")
        return alignment_count
    
    def preference(self, t1 : list[str], t2 : list[str]) -> int:
        """Obtains simulated user's preference between trajectories t1 and t2. 

        Args:
            t1 (list[str]): List of terrain types associated with trajectory one.
            t2 (list[str]): List of terrain types associated with trajectory two. 

        Returns:
            int: Simulated choice (0 or 1) between trajectories. 
        """
        if self.check_distribution(t1) > self.check_distribution(t2):
            return 1 # t1 had more out of order terrains, pick t2
        elif self.check_distribution(t1) < self.check_distribution(t2):
            return 0 # pick t1
        else:
            return random.choice([0, 1]) # if equal, random choice


    
    def respond(self, queries) -> list:
        """
        Interactively asks for the user's responses to the given queries.
        
        Args:
            queries (Query or List[Query]): A query or a list of queries for which the user's response(s)
                is/are requested.
                
        Returns:
            List: A list of user responses where each response corresponds to the query in the :py:attr:`queries`.
                :Note: The return type is always a list, even if the input is a single query.
        """
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            # query.slate[0] gives a trajectory.
            responses.append(self.preference(query.slate[0].features[1], query.slate[1].features[1]))
#            responses.append(query.visualize(self.delay))
        return responses

# from Félix Therrien
from typing import Literal

from pymatgen.analysis.local_env import (
    VoronoiNN,
    _handle_disorder,
    _is_in_targets,
)
from pymatgen.core.structure import Structure

on_disorder_options = Literal[
    "take_majority_strict", "take_majority_drop", "take_max_species", "error"
]


class PatchedVoronoiNN(VoronoiNN):
    def get_cn(
        self,
        structure: Structure,
        n: int,
        use_weights: bool = False,
        on_disorder: on_disorder_options = "take_majority_strict",
    ) -> float:
        """
        Get coordination number, CN, of site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True) to use weights for computing the coordination
                number or not (False, default: each coordinated site has equal weight).
            on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
                What to do when encountering a disordered structure. 'error' will raise ValueError.
                'take_majority_strict' will use the majority specie on each site and raise
                ValueError if no majority exists. 'take_max_species' will use the first max specie
                on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
                will raise ValueError, while 'take_majority_drop' ignores this site altogether and
                'take_max_species' will use Fe as the site specie.

        Returns:
            cn (int or float): coordination number.
        """
        structure = _handle_disorder(structure, on_disorder)
        # siw = self.get_nn_info(structure, n)
        siw = self.get_nn_info(structure, n, weights_only=True)
        return sum(e["weight"] for e in siw) if use_weights else len(siw)

    def _extract_nn_info(self, structure: Structure, nns, weights_only=False):
        """Given Voronoi NNs, extract the NN info in the form needed by NearestNeighbors

        Args:
            structure (Structure): Structure being evaluated
            nns ([dicts]): Nearest neighbor information for a structure
            weights_only (bool): Whether to only return the weights
        Returns:
            (list of tuples (Site, array, float)): See nn_info
        """
        # Get the target information
        targets = (
            structure.composition.elements if self.targets is None else self.targets
        )

        # Extract the NN info
        siw = []
        max_weight = max(nn[self.weight] for nn in nns.values())
        for nstats in nns.values():
            site = nstats["site"]
            if nstats[self.weight] > self.tol * max_weight and _is_in_targets(
                site, targets
            ):
                if weights_only:
                    nn_info = {"weight": nstats[self.weight] / max_weight}
                else:
                    weights_only = {
                        "site": site,
                        "image": self._get_image(structure, site),
                        "weight": nstats[self.weight] / max_weight,
                        "site_index": self._get_original_site(structure, site),
                    }
                    if self.extra_nn_info:
                        # Add all the information about the site
                        poly_info = nstats
                        del poly_info["site"]
                        nn_info["poly_info"] = poly_info
                siw.append(nn_info)

        return siw

    def get_all_nn_info(self, structure):
        """
        Args:
            structure (Structure): input structure.

        Returns:
            All nn info for all sites.
        """
        all_voro_cells = self.get_all_voronoi_polyhedra(structure)
        return [self._extract_nn_info(structure, cell) for cell in all_voro_cells]

    def get_nn_info(self, structure: Structure, n: int, weights_only=False):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure
        using Voronoi decomposition.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor
                sites.
            weights_only (bool): Whether to only return the weights

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        # Run the tessellation
        nns = self.get_voronoi_polyhedra(structure, n)

        # Extract the NN info
        return self._extract_nn_info(structure, nns, weights_only)

    def get_cn_dict(self, structure: Structure, n: int, use_weights: bool = False):
        """
        Get coordination number, CN, of each element bonded to site with index n in structure

        Args:
            structure (Structure): input structure
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).

        Returns:
            cn (dict): dictionary of CN of each element bonded to site
        """
        siw = self.get_nn_info(structure, n)

        cn_dict = {}
        for idx in siw:
            site_element = idx["site"].species_string
            if site_element not in cn_dict:
                if use_weights:
                    cn_dict[site_element] = idx["weight"]
                else:
                    cn_dict[site_element] = 1
            else:
                if use_weights:
                    cn_dict[site_element] += idx["weight"]
                else:
                    cn_dict[site_element] += 1
        return cn_dict

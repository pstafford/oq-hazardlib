# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Source model XML Writer
"""

import os
from openquake.baselib.general import CallableDict
from openquake.baselib.node import Node
from openquake.hazardlib import nrml

obj_to_node = CallableDict(lambda obj: obj.__class__.__name__)


def build_area_source_geometry(area_source):
    """
    Returns the area source geometry as a Node

    :param area_source:
        Area source model as an instance of the :class:
        `openquake.hazardlib.source.area.AreaSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    geom_str = ["%s %s" % lonlat for lonlat in
                zip(area_source.polygon.lons, area_source.polygon.lats)]
    poslist_node = Node("gml:posList", text=geom_str)
    linear_ring_node = Node("gml:LinearRing", nodes=[poslist_node])
    exterior_node = Node("gml:exterior", nodes=[linear_ring_node])
    polygon_node = Node("gml:Polygon", nodes=[exterior_node])
    upper_depth_node = Node(
        "upperSeismoDepth", text=area_source.upper_seismogenic_depth)
    lower_depth_node = Node(
        "lowerSeismoDepth", text=area_source.lower_seismogenic_depth)
    return Node(
        "areaGeometry", {'discretization': area_source.area_discretization},
        nodes=[polygon_node, upper_depth_node, lower_depth_node])


def build_point_source_geometry(point_source):
    """
    Returns the poing source geometry as a Node

    :param point_source:
        Point source model as an instance of the :class:
        `openquake.hazardlib.source.point.PointSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    xy = point_source.location.x, point_source.location.y
    pos_node = Node("gml:pos", text=xy)
    point_node = Node("gml:Point", nodes=[pos_node])
    upper_depth_node = Node(
        "upperSeismoDepth", text=point_source.upper_seismogenic_depth)
    lower_depth_node = Node(
        "lowerSeismoDepth", text=point_source.lower_seismogenic_depth)
    return Node(
        "pointGeometry",
        nodes=[point_node, upper_depth_node, lower_depth_node])


def build_linestring_node(line, with_depth=False):
    """
    Parses a line to a Node class

    :param line:
        Line as instance of :class:`openquake.hazardlib.geo.line.Line`
    :param bool with_depth:
        Include the depth values (True) or not (False):
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    if with_depth:
        geom_str = ["%s %s %s" % (p.x, p.y, p.z) for p in line.points]
    else:
        geom_str = ["%s %s" % (p.x, p.y) for p in line.points]
    poslist_node = Node("gml:posList", text=geom_str)
    return Node("gml:LineString", nodes=[poslist_node])


def build_simple_fault_geometry(fault_source):
    """
    Returns the simple fault source geometry as a Node

    :param fault_source:
        Simple fault source model as an instance of the :class:
        `openquake.hazardlib.source.simple_fault.SimpleFaultSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    linestring_node = build_linestring_node(fault_source.fault_trace,
                                            with_depth=False)
    dip_node = Node("dip", text=fault_source.dip)
    upper_depth_node = Node(
        "upperSeismoDepth", text=fault_source.upper_seismogenic_depth)
    lower_depth_node = Node(
        "lowerSeismoDepth", text=fault_source.lower_seismogenic_depth)
    return Node("simpleFaultGeometry",
                nodes=[linestring_node, dip_node, upper_depth_node,
                       lower_depth_node])


def build_complex_fault_geometry(fault_source):
    """
    Returns the complex fault source geometry as a Node

    :param fault_source:
        Complex fault source model as an instance of the :class:
        `openquake.hazardlib.source.complex_fault.ComplexFaultSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    num_edges = len(fault_source.edges)
    edge_nodes = []
    for iloc, edge in enumerate(fault_source.edges):
        if iloc == 0:
            # Top Edge
            node_name = "faultTopEdge"

        elif iloc == (num_edges - 1):
            # Bottom edge
            node_name = "faultBottomEdge"
        else:
            # Intermediate edge
            node_name = "intermediateEdge"
        edge_nodes.append(
            Node(node_name,
                 nodes=[build_linestring_node(edge, with_depth=True)]))
    return Node("complexFaultGeometry", nodes=edge_nodes)


@obj_to_node.add('EvenlyDiscretizedMFD')
def build_evenly_discretised_mfd(mfd):
    """
    Returns the evenly discretized MFD as a Node

    :param mfd:
        MFD as instance of :class:
        `openquake.hazardlib.mfd.evenly_discretized.EvenlyDiscretizedMFD`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    occur_rates = Node("occurRates", text=mfd.occurrence_rates)
    return Node("incrementalMFD",
                {"binWidth": mfd.bin_width, "minMag": mfd.min_mag},
                nodes=[occur_rates])


@obj_to_node.add('TruncatedGRMFD')
def build_truncated_gr_mfd(mfd):
    """
    Parses the truncated Gutenberg Richter MFD as a Node

    :param mfd:
        MFD as instance of :class:
        `openquake.hazardlib.mfd.truncated_gr.TruncatedGRMFD`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    return Node("truncGutenbergRichterMFD",
                {"aValue": mfd.a_val, "bValue": mfd.b_val,
                 "minMag": mfd.min_mag, "maxMag": mfd.max_mag})


@obj_to_node.add('ArbitraryMFD')
def build_arbitrary_mfd(mfd):
    """
    Parses the arbitrary MFD as a Node

    :param mfd:
        MFD as instance of :class:
        `openquake.hazardlib.mfd.arbitrary.ArbitraryMFD`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    magnitudes = Node("magnitudes", text=mfd.magnitudes)
    occur_rates = Node("occurRates", text=mfd.occurrence_rates)
    return Node("arbitraryMFD", nodes=[magnitudes, occur_rates])


@obj_to_node.add("YoungsCoppersmith1985MFD")
def build_youngs_coppersmith_mfd(mfd):
    """
    Parses the Youngs & Coppersmith MFD as a node. Note that the MFD does
    not hold the total moment rate, but only the characteristic rate. Therefore
    the node is written to the characteristic rate version regardless of
    whether or not it was originally created from total moment rate

    :param mfd:
        MFD as instance of :class:
        `openquake.hazardlib.mfd.youngs_coppersmith_1985.
        YoungsCoppersmith1985MFD`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    return Node("YoungsCoppersmithMFD",
                {"minMag": mfd.min_mag, "bValue": mfd.b_val,
                 "characteristicMag": mfd.char_mag,
                 "characteristicRate": mfd.char_rate,
                 "binWidth": mfd.bin_width})


def build_nodal_plane_dist(npd):
    """
    Returns the nodal plane distribution as a Node instance

    :param npd:
        Nodal plane distribution as instance of :class:
        `openquake.hazardlib.pmf.PMF`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    npds = []
    for prob, npd in npd.data:
        nodal_plane = Node(
            "nodalPlane", {"dip": npd.dip, "probability": prob,
                           "strike": npd.strike, "rake": npd.rake})
        npds.append(nodal_plane)
    return Node("nodalPlaneDist", nodes=npds)


def build_hypo_depth_dist(hdd):
    """
    Returns the hypocentral depth distribution as a Node instance

    :param hdd:
        Hypocentral depth distribution as an instance of :class:
        `openquake.hzardlib.pmf.PMF`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    hdds = []
    for (prob, depth) in hdd.data:
        hdds.append(
            Node("hypoDepth", {"depth": depth, "probability": prob}))
    return Node("hypoDepthDist", nodes=hdds)


def get_distributed_seismicity_source_nodes(source):
    """
    Returns list of nodes of attributes common to all distributed seismicity
    source classes

    :param source:
        Seismic source as instance of :class:
        `openquake.hazardlib.source.area.AreaSource` or :class:
        `openquake.hazardlib.source.point.PointSource`
    :returns:
        List of instances of :class:`openquake.baselib.node.Node`
    """
    source_nodes = []
    #  parse msr
    source_nodes.append(
        Node("magScaleRel",
             text=source.magnitude_scaling_relationship.__class__.__name__))
    # Parse aspect ratio
    source_nodes.append(
        Node("ruptAspectRatio", text=source.rupture_aspect_ratio))
    # Parse MFD
    source_nodes.append(obj_to_node(source.mfd))
    # Parse nodal plane distribution
    source_nodes.append(
        build_nodal_plane_dist(source.nodal_plane_distribution))
    # Parse hypocentral depth distribution
    source_nodes.append(
        build_hypo_depth_dist(source.hypocenter_distribution))
    return source_nodes


def build_hypo_list_node(hypo_list):
    """
    :param hypo_list:
       an array of shape (N, 3) with columns (alongStrike, downDip, weight)
    :returns:
        a hypoList node containing N hypo nodes
    """
    hypolist = Node('hypoList', {})
    for row in hypo_list:
        n = Node(
            'hypo', dict(alongStrike=row[0], downDip=row[1], weight=row[2]))
        hypolist.append(n)
    return hypolist


def build_slip_list_node(slip_list):
    """
    :param slip_list:
       an array of shape (N, 2) with columns (slip, weight)
    :returns:
        a hypoList node containing N slip nodes
    """
    sliplist = Node('slipList', {})
    for row in slip_list:
        sliplist.append(
            Node('slip', dict(weight=row[1]), row[0]))
    return sliplist


def get_fault_source_nodes(source):
    """
    Returns list of nodes of attributes common to all fault source classes

    :param source:
        Fault source as instance of :class:
        `openquake.hazardlib.source.simple_fault.SimpleFaultSource` or :class:
        `openquake.hazardlib.source.complex_fault.ComplexFaultSource`
    :returns:
        List of instances of :class:`openquake.baselib.node.Node`
    """
    source_nodes = []
    #  parse msr
    source_nodes.append(
        Node(
            "magScaleRel",
            text=source.magnitude_scaling_relationship.__class__.__name__))
    # Parse aspect ratio
    source_nodes.append(
        Node("ruptAspectRatio", text=source.rupture_aspect_ratio))
    # Parse MFD
    source_nodes.append(obj_to_node(source.mfd))
    # Parse Rake
    source_nodes.append(Node("rake", text=source.rake))
    if len(getattr(source, 'hypo_list', [])):
        source_nodes.append(build_hypo_list_node(source.hypo_list))
    if len(getattr(source, 'slip_list', [])):
        source_nodes.append(build_slip_list_node(source.slip_list))
    return source_nodes


def get_source_attributes(source):
    """
    Retreives a dictionary of source attributes from the source class

    :param source:
        Seismic source as instance of :class:
        `openquake.hazardlib.source.base.BaseSeismicSource`
    :returns:
        Dictionary of source attributes
    """
    return {"id": source.source_id,
            "name": source.name,
            "tectonicRegion": source.tectonic_region_type}


@obj_to_node.add('AreaSource')
def build_area_source_node(area_source):
    """
    Parses an area source to a Node class

    :param area_source:
        Area source as instance of :class:
        `openquake.hazardlib.source.area.AreaSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    # parse geometry
    source_nodes = [build_area_source_geometry(area_source)]
    # parse common distributed attributes
    source_nodes.extend(get_distributed_seismicity_source_nodes(area_source))
    return Node(
        "areaSource", get_source_attributes(area_source), nodes=source_nodes)


@obj_to_node.add('CharacteristicFaultSource')
def build_characteristic_fault_source_node(source):
    source_nodes = [obj_to_node(source.mfd)]
    source_nodes.append(Node("rake", text=source.rake))
    surface_node = Node('surface', nodes=source.surface.surface_nodes)
    source_nodes.append(surface_node)
    return Node('characteristicFaultSource',
                get_source_attributes(source),
                nodes=source_nodes)


@obj_to_node.add('NonParametricSeismicSource')
def build_nonparametric_source_node(source):
    rup_nodes = []
    for rup, pmf in source.data:
        probs = [prob for (prob, no) in pmf.data]
        rup_nodes.append(build_rupture_node(rup, probs))
    return Node('nonParametricSeismicSource',
                get_source_attributes(source), nodes=rup_nodes)


def build_rupture_node(rupt, probs_occur):
    """
    :param rupt: a hazardlib rupture object
    :param probs_occur: a list of floats with sum 1
    """
    h = rupt.hypocenter
    hp_dict = dict(lon=h.longitude, lat=h.latitude, depth=h.depth)
    rupt_nodes = [Node('magnitude', {}, rupt.mag),
                  Node('rake', {}, rupt.rake),
                  Node('hypocenter', hp_dict)]
    rupt_nodes.extend(rupt.surface.surface_nodes)
    geom = rupt.surface.surface_nodes[0].tag
    if len(rupt.surface.surface_nodes) > 1:
        name = 'multiPlanesRupture'
    elif geom == 'planarSurface':
        name = 'singlePlaneRupture'
    elif geom == 'simpleFaultGeometry':
        name = 'simpleFaultRupture'
    elif geom == 'complexFaultGeometry':
        name = 'complexFaultRupture'
    return Node(name, {'probs_occur': probs_occur}, nodes=rupt_nodes)


@obj_to_node.add('PointSource')
def build_point_source_node(point_source):
    """
    Parses a point source to a Node class

    :param point_source:
        Point source as instance of :class:
        `openquake.hazardlib.source.point.PointSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`

    """
    # parse geometry
    source_nodes = [build_point_source_geometry(point_source)]
    # parse common distributed attributes
    source_nodes.extend(get_distributed_seismicity_source_nodes(point_source))
    return Node("pointSource",
                get_source_attributes(point_source),
                nodes=source_nodes)


@obj_to_node.add('SimpleFaultSource')
def build_simple_fault_source_node(fault_source):
    """
    Parses a simple fault source to a Node class

    :param fault_source:
        Simple fault source as instance of :class:
        `openquake.hazardlib.source.simple_fault.SimpleFaultSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    # Parse geometry
    source_nodes = [build_simple_fault_geometry(fault_source)]
    # Parse common fault source attributes
    source_nodes.extend(get_fault_source_nodes(fault_source))
    return Node("simpleFaultSource",
                get_source_attributes(fault_source),
                nodes=source_nodes)


@obj_to_node.add('ComplexFaultSource')
def build_complex_fault_source_node(fault_source):
    """
    Parses a complex fault source to a Node class

    :param fault_source:
        Simple fault source as instance of :class:
        `openquake.hazardlib.source.complex_fault.ComplexFaultSource`
    :returns:
        Instance of :class:`openquake.baselib.node.Node`
    """
    # Parse geometry
    source_nodes = [build_complex_fault_geometry(fault_source)]
    # Parse common fault source attributes
    source_nodes.extend(get_fault_source_nodes(fault_source))
    return Node("complexFaultSource",
                get_source_attributes(fault_source),
                nodes=source_nodes)


@obj_to_node.add('SourceGroup')
def build_source_group(source_group):
    source_nodes = [obj_to_node(src) for src in source_group.sources]
    attrs = dict(tectonicRegion=source_group.trt)
    if source_group.name:
        attrs['name'] = source_group.name
    if source_group.src_interdep:
        attrs['src_interdep'] = source_group.src_interdep
    if source_group.rup_interdep:
        attrs['rup_interdep'] = source_group.rup_interdep
    if source_group.srcs_weights:
        attrs['srcs_weights'] = ' '.join(map(str, source_group.srcs_weights))
    return Node('sourceGroup', attrs, nodes=source_nodes)


# ##################### generic source model writer ####################### #

def write_source_model(dest, groups, name=None):
    """
    Writes a source model to XML.

    :param str dest:
        Destination path
    :param list groups:
        Source model as list of SourceGroups
    :param str name:
        Name of the source model (if missing, extracted from the filename)
    """
    name = name or os.path.splitext(os.path.basename(dest))[0]
    nodes = list(map(obj_to_node, sorted(groups)))
    source_model = Node("sourceModel", {"name": name}, nodes=nodes)
    with open(dest, 'wb') as f:
        nrml.write([source_model], f, '%s')
    return dest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

class Preprocess(object):
    def raw2train(self, path_to_data, target_rename=False, target_remove=False, save=False, file_name='joint_renamed_v2', save_intermediate=False, return_intermediate=False, intermediate_names=['gm_common_renamed', 'k_common_renamed', 'ppi_undirected']):
        
        # Read raw data
        path = Path(path_to_data)
        ppi = pd.read_csv(path / 'biogrid.hc.tsv', sep='\t', header=None)
        gm = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)
        k = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_K562.tsv', sep='\t', header=None)

        # Rename headers
        gm.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        k.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        ppi.columns = ['source', 'target', 'type', 'dummy']
        ppi = ppi[['source', 'target', 'type']]

        print('Raw data:')
        print(f'Shape of GM12878: {gm.shape}')
        print(f'Shape of K562: {k.shape}')
        print(f'Shape of PPI: {ppi.shape}')
        print('-----------------------------------------')
        # Extract nodes
        ppi_nodes = set(ppi['source']).union(set(ppi['target']))
        print(len(ppi_nodes))
        gm_nodes = set(gm['target'])
        k_nodes = set(k['target'])

        # Transform PPI to undirected graph by swapping its source and target
        ppi_reverse = ppi[['target', 'source', 'type']]
        ppi_reverse.columns = ['source', 'target', 'type']
        ppi_undirected = pd.concat([ppi, ppi_reverse])
        ppi_undirected.sort_values(['source', 'target'], ascending=True, inplace=True)
        ppi_undirected.reset_index(inplace=True)
        ppi_undirected = ppi_undirected[['source', 'target', 'type']]
        ppi_undirected['weight'] = 'NA'
        
        # Lookup table for TFs
        common_tf_df = self.raw2tf(path_to_data, option='intersection')
        all_tf_df = self.raw2tf(path_to_data, option='union')
        xor_tf_df = self.raw2tf(path_to_data, option='xor')
        common_tf = set(common_tf_df['tf'])
        all_tf = set(all_tf_df['tf'])
        xor_tf = set(xor_tf_df['tf'])

        # Process source nodes 
        # Filter out source nodes that belong to common_tf
        gm_tf2gene = gm[gm['source'].isin(common_tf)]
        k_tf2gene = k[k['source'].isin(common_tf)]

        # Clean up
        gm_tf2gene.reset_index()
        gm_tf2gene = gm_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']]
        k_tf2gene.reset_index()
        k_tf2gene = k_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']]

        # rename ALL source nodes (add '_gm' or '_k')
        # ALL source nodes are TFs
        gm_tf2gene['source_renamed'] = gm_tf2gene['source'].map(lambda x: x + '_gm')
        k_tf2gene['source_renamed'] = k_tf2gene['source'].map(lambda x: x + '_k')
        gm_tf2gene['source'] = gm_tf2gene['source_renamed']
        k_tf2gene['source'] = k_tf2gene['source_renamed']
        gm_tf2gene.drop(['source_renamed'], axis=1, inplace=True)
        k_tf2gene.drop(['source_renamed'], axis=1, inplace=True)


        # Process target nodes
        # At this stage, all the `source` are common TFs, next steps is to identify target nodes
        # 1. gene -- no operation
        # 2. TF but not common_tf -- remove
        # 3. TF and part of common_tf -- rename?

        if target_rename:
            # Rename target nodes which are TFs AND part of common_tf
            gm_tf2gene['target_renamed'] = gm_tf2gene['target'].map(lambda x: x + '_gm' if x in common_tf else x)
            k_tf2gene['target_renamed'] = k_tf2gene['target'].map(lambda x: x + '_k' if x in common_tf else x)
            # Clean up the DataFrame and save
            gm_tf2gene.drop(['target'], axis=1, inplace=True)
            k_tf2gene.drop(['target'], axis=1, inplace=True)
            
            gm_tf2gene['target'] = gm_tf2gene['target_renamed']
            k_tf2gene['target'] = k_tf2gene['target_renamed']
            gm_tf2gene.drop(['target_renamed'], axis=1, inplace=True)
            k_tf2gene.drop(['target_renamed'], axis=1, inplace=True)

        if target_remove:
            # Remove target nodes which are TFs BUT NOT part of common_tf
            gm_tf2gene = gm_tf2gene[~gm_tf2gene['target'].isin(xor_tf)]
            k_tf2gene = k_tf2gene[~k_tf2gene['target'].isin(xor_tf)]

        

        # Save intermediate DataFrame
        if save_intermediate:
            gm_tf2gene.to_csv(path / f'{intermediate_names[0]}.csv', index=False)
            k_tf2gene.to_csv(path / f'{intermediate_names[1]}.csv', index=False)
            ppi_undirected.to_csv(path / f'{intermediate_names[2]}.csv', index=False)

        # Merge
        print(f'After processing:')
        print(f'Number of GM12878 edges: {gm_tf2gene.shape[0]}')
        print(f'Number of K562 edges: {k_tf2gene.shape[0]}')
        print(f'Number of PPI edges (Undirectional): {int(ppi_undirected.shape[0] / 2)}')
        print('--------')
        print(f'Number of TFs: {len(common_tf)}')
        print('-----------------------------------------')

        ppi_undirected['cell_type'] = 'NA'
        needed_cols = ['cell_type', 'source', 'target', 'type', 'weight']

        merged_renamed = pd.concat([gm_tf2gene[needed_cols], k_tf2gene[needed_cols], ppi_undirected])
        merged_renamed['cell_type'] = merged_renamed['cell_type'].astype(object)
        merged_renamed.reset_index(inplace=True)
        merged_renamed.drop_duplicates(inplace=True)

        # Save training data
        if save:
            merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']].to_csv(path / f'{file_name}.csv', index=False)
        
        if return_intermediate:
            return merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']], gm_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']], k_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']], ppi_undirected[['cell_type', 'source', 'target', 'type', 'weight']]

        return merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']]


    def raw2tf(self, path_to_data, option, save=False):
        # Read raw data
        path = Path(path_to_data)
        gm = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)
        k = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_K562.tsv', sep='\t', header=None)
        gm.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        k.columns = ['cell_type', 'source', 'target', 'type', 'weight']

        # Extract TFs
        gm12878_tf = set(gm['source'])
        k562_tf = set(k['source'])
        if option == 'intersection':
            common_tf = set(k562_tf.intersection(gm12878_tf))
            return pd.DataFrame(common_tf, columns=['tf'])
        elif option == 'union':
            all_tf = set(k562_tf.union(k562_tf))
            return pd.DataFrame(all_tf, columns=['tf'])
        elif option == 'xor':
            common_tf = set(k562_tf.intersection(gm12878_tf))
            all_tf = set(k562_tf.union(k562_tf))
            xor_tf = all_tf.difference(common_tf)
            return pd.DataFrame(xor_tf, columns=['tf'])
        else:
            raise InvalidOptionException(f'{option} is not a valid option')
    
    
class Feature(object):
    def onehot_names(self, df):
        nodes = set(df['source']).union(set(df['target']))
        nodes = list(nodes)
        nodes.sort()
        node_df = pd.DataFrame(nodes)

        # Create one-hot encoding df
        onehot_df = pd.get_dummies(node_df, prefix='node')
        onehot_df.index = nodes
        # onehot_df['feature'] = onehot_df.index.map(lambda x: onehot_df.loc[x].to_numpy())
        # onehot_df = onehot_df[['feature']]
        return onehot_df

    def node2deg(self, df):
        nx_graph = nx.from_pandas_edgelist(df[['source', 'target', 'weight']], 'source', 'target', edge_attr='weight', create_using=nx.DiGraph)
        nodes = nx_graph.nodes()
        return pd.DataFrame(list(nx_graph.degree(nodes)), index=nodes)[[1]]

    def adjacentTFs(self, df, common_tf):
        common_tf_k = [_tf + '_k' for _tf in common_tf]
        common_tf_gm = [_tf + '_gm' for _tf in common_tf]
        common_tf = common_tf_k + common_tf_gm
        common_tf.sort()

        d = dict()
        for i, tf in enumerate(common_tf):
            d[tf] = i

        nodes = list(set(df['source']).union(set(df['target'])))

        def node2neighbors(node, df=df):
            feature = np.array([0] * len(common_tf))
            a = df['target'] == node
            b = df['source'].isin(common_tf)
            regulators = list(df[a & b]['source'])
            linked_tf_pos = list(map(lambda tf: d[tf], regulators))
            feature[linked_tf_pos] = 1
            return feature

        features = dict(zip(nodes, map(node2neighbors, nodes)))
        feature_df = pd.DataFrame.from_dict(features, orient='index')
        return feature_df


class PreprocessForTrail(object):
    def raw2train(self, path_to_data, target_rename=False, target_remove=False, save=False, file_name='2gm_renamed', save_intermediate=False, return_intermediate=False, intermediate_names=['gm_common_renamed', 'k_common_renamed', 'ppi_undirected']):
        
        # Read raw data
        path = Path(path_to_data)
        ppi = pd.read_csv(path / 'biogrid.hc.tsv', sep='\t', header=None)
        gm = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)
        k = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)

        # Rename headers
        gm.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        k.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        ppi.columns = ['source', 'target', 'type', 'dummy']
        ppi = ppi[['source', 'target', 'type']]

        print('Raw data:')
        print(f'Shape of GM12878: {gm.shape}')
        print(f'Shape of K562: {k.shape}')
        print(f'Shape of PPI: {ppi.shape}')
        print('-----------------------------------------')
        # Extract nodes
        ppi_nodes = set(ppi['source']).union(set(ppi['target']))
        print(len(ppi_nodes))
        gm_nodes = set(gm['target'])
        k_nodes = set(k['target'])

        # Transform PPI to undirected graph by swapping its source and target
        ppi_reverse = ppi[['target', 'source', 'type']]
        ppi_reverse.columns = ['source', 'target', 'type']
        ppi_undirected = pd.concat([ppi, ppi_reverse])
        ppi_undirected.sort_values(['source', 'target'], ascending=True, inplace=True)
        ppi_undirected.reset_index(inplace=True)
        ppi_undirected = ppi_undirected[['source', 'target', 'type']]
        ppi_undirected['weight'] = 'NA'
        
        # Lookup table for TFs
        common_tf_df = self.raw2tf(path_to_data, option='intersection')
        all_tf_df = self.raw2tf(path_to_data, option='union')
        xor_tf_df = self.raw2tf(path_to_data, option='xor')
        common_tf = set(common_tf_df['tf'])
        all_tf = set(all_tf_df['tf'])
        xor_tf = set(xor_tf_df['tf'])

        # Process source nodes 
        # Filter out source nodes that belong to common_tf
        gm_tf2gene = gm[gm['source'].isin(common_tf)]
        k_tf2gene = k[k['source'].isin(common_tf)]

        # Clean up
        gm_tf2gene.reset_index()
        gm_tf2gene = gm_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']]
        k_tf2gene.reset_index()
        k_tf2gene = k_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']]

        # rename ALL source nodes (add '_gm' or '_k')
        # ALL source nodes are TFs
        gm_tf2gene['source_renamed'] = gm_tf2gene['source'].map(lambda x: x + '_gm1')
        k_tf2gene['source_renamed'] = k_tf2gene['source'].map(lambda x: x + '_gm2')
        gm_tf2gene['source'] = gm_tf2gene['source_renamed']
        k_tf2gene['source'] = k_tf2gene['source_renamed']
        gm_tf2gene.drop(['source_renamed'], axis=1, inplace=True)
        k_tf2gene.drop(['source_renamed'], axis=1, inplace=True)


        # Process target nodes
        # At this stage, all the `source` are common TFs, next steps is to identify target nodes
        # 1. gene -- no operation
        # 2. TF but not common_tf -- remove
        # 3. TF and part of common_tf -- rename?

        if target_rename:
            # Rename target nodes which are TFs AND part of common_tf
            gm_tf2gene['target_renamed'] = gm_tf2gene['target'].map(lambda x: x + '_gm1' if x in common_tf else x)
            k_tf2gene['target_renamed'] = k_tf2gene['target'].map(lambda x: x + '_gm2' if x in common_tf else x)
            # Clean up the DataFrame and save
            gm_tf2gene.drop(['target'], axis=1, inplace=True)
            k_tf2gene.drop(['target'], axis=1, inplace=True)
            
            gm_tf2gene['target'] = gm_tf2gene['target_renamed']
            k_tf2gene['target'] = k_tf2gene['target_renamed']
            gm_tf2gene.drop(['target_renamed'], axis=1, inplace=True)
            k_tf2gene.drop(['target_renamed'], axis=1, inplace=True)

        if target_remove:
            # Remove target nodes which are TFs BUT NOT part of common_tf
            gm_tf2gene = gm_tf2gene[~gm_tf2gene['target'].isin(xor_tf)]
            k_tf2gene = k_tf2gene[~k_tf2gene['target'].isin(xor_tf)]

        

        # Save intermediate DataFrame
        if save_intermediate:
            gm_tf2gene.to_csv(path / f'{intermediate_names[0]}.csv', index=False)
            k_tf2gene.to_csv(path / f'{intermediate_names[1]}.csv', index=False)
            ppi_undirected.to_csv(path / f'{intermediate_names[2]}.csv', index=False)

        # Merge
        print(f'After processing:')
        print(f'Number of GM12878 edges: {gm_tf2gene.shape[0]}')
        print(f'Number of K562 edges: {k_tf2gene.shape[0]}')
        print(f'Number of PPI edges (Undirectional): {int(ppi_undirected.shape[0] / 2)}')
        print('--------')
        print(f'Number of TFs: {len(common_tf)}')
        print('-----------------------------------------')

        ppi_undirected['cell_type'] = 'NA'
        needed_cols = ['cell_type', 'source', 'target', 'type', 'weight']

        merged_renamed = pd.concat([gm_tf2gene[needed_cols], k_tf2gene[needed_cols], ppi_undirected])
        merged_renamed['cell_type'] = merged_renamed['cell_type'].astype(object)
        merged_renamed.reset_index(inplace=True)
        merged_renamed.drop_duplicates(inplace=True)

        # Save training data
        if save:
            merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']].to_csv(path / f'{file_name}.csv', index=False)
        
        if return_intermediate:
            return merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']], gm_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']], k_tf2gene[['cell_type', 'source', 'target', 'type', 'weight']], ppi_undirected[['cell_type', 'source', 'target', 'type', 'weight']]

        return merged_renamed[['cell_type', 'source', 'target', 'type', 'weight']]


    def raw2tf(self, path_to_data, option, save=False):
        # Read raw data
        path = Path(path_to_data)
        gm = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)
        k = pd.read_csv(path / 'EC-003-NET.edgeList_TSS_GM12878.tsv', sep='\t', header=None)
        gm.columns = ['cell_type', 'source', 'target', 'type', 'weight']
        k.columns = ['cell_type', 'source', 'target', 'type', 'weight']

        # Extract TFs
        gm12878_tf = set(gm['source'])
        k562_tf = set(k['source'])
        if option == 'intersection':
            common_tf = set(k562_tf.intersection(gm12878_tf))
            return pd.DataFrame(common_tf, columns=['tf'])
        elif option == 'union':
            all_tf = set(k562_tf.union(k562_tf))
            return pd.DataFrame(all_tf, columns=['tf'])
        elif option == 'xor':
            common_tf = set(k562_tf.intersection(gm12878_tf))
            all_tf = set(k562_tf.union(k562_tf))
            xor_tf = all_tf.difference(common_tf)
            return pd.DataFrame(xor_tf, columns=['tf'])
        else:
            raise InvalidOptionException(f'{option} is not a valid option')
    
    
class Feature(object):
    def onehot_names(self, df):
        nodes = set(df['source']).union(set(df['target']))
        nodes = list(nodes)
        nodes.sort()
        node_df = pd.DataFrame(nodes)

        # Create one-hot encoding df
        onehot_df = pd.get_dummies(node_df, prefix='node')
        onehot_df.index = nodes
        # onehot_df['feature'] = onehot_df.index.map(lambda x: onehot_df.loc[x].to_numpy())
        # onehot_df = onehot_df[['feature']]
        return onehot_df

    def node2deg(self, df):
        nx_graph = nx.from_pandas_edgelist(df[['source', 'target', 'weight']], 'source', 'target', edge_attr='weight', create_using=nx.DiGraph)
        nodes = nx_graph.nodes()
        return pd.DataFrame(list(nx_graph.degree(nodes)), index=nodes)[[1]]

    def adjacentTFs(self, df, common_tf):
        common_tf_k = [_tf + '_gm1' for _tf in common_tf]
        common_tf_gm = [_tf + '_gm2' for _tf in common_tf]
        common_tf = common_tf_k + common_tf_gm
        common_tf.sort()

        d = dict()
        for i, tf in enumerate(common_tf):
            d[tf] = i

        nodes = list(set(df['source']).union(set(df['target'])))

        def node2neighbors(node, df=df):
            feature = np.array([0] * len(common_tf))
            a = df['target'] == node
            b = df['source'].isin(common_tf)
            regulators = list(df[a & b]['source'])
            linked_tf_pos = list(map(lambda tf: d[tf], regulators))
            feature[linked_tf_pos] = 1
            return feature

        features = dict(zip(nodes, map(node2neighbors, nodes)))
        feature_df = pd.DataFrame.from_dict(features, orient='index')
        return feature_df


class InvalidOptionException(Exception):
    pass




# path_to_data = 'C:/Users/mukun/Desktop/research/graph_embedding/data'
# print(raw2train(path_to_data)) 
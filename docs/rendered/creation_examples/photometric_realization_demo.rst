Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f175a203ca0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>27.361561</td>
          <td>0.706672</td>
          <td>26.759504</td>
          <td>0.172930</td>
          <td>25.873434</td>
          <td>0.070500</td>
          <td>25.167711</td>
          <td>0.061607</td>
          <td>24.643274</td>
          <td>0.074108</td>
          <td>23.941693</td>
          <td>0.089792</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.367733</td>
          <td>0.286550</td>
          <td>26.586375</td>
          <td>0.131704</td>
          <td>26.485924</td>
          <td>0.193962</td>
          <td>26.157387</td>
          <td>0.271188</td>
          <td>25.725540</td>
          <td>0.399184</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.475283</td>
          <td>0.762526</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.674503</td>
          <td>0.326498</td>
          <td>25.938694</td>
          <td>0.121386</td>
          <td>24.979312</td>
          <td>0.099623</td>
          <td>24.185369</td>
          <td>0.111159</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>31.446153</td>
          <td>4.017180</td>
          <td>27.820620</td>
          <td>0.409517</td>
          <td>27.092678</td>
          <td>0.202827</td>
          <td>26.265541</td>
          <td>0.160883</td>
          <td>25.734470</td>
          <td>0.190954</td>
          <td>26.045375</td>
          <td>0.507915</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.759574</td>
          <td>0.207216</td>
          <td>26.053520</td>
          <td>0.093912</td>
          <td>25.936723</td>
          <td>0.074560</td>
          <td>25.790478</td>
          <td>0.106679</td>
          <td>25.650295</td>
          <td>0.177834</td>
          <td>25.065480</td>
          <td>0.235356</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.720209</td>
          <td>0.892913</td>
          <td>26.389074</td>
          <td>0.125834</td>
          <td>25.418622</td>
          <td>0.047093</td>
          <td>25.057015</td>
          <td>0.055843</td>
          <td>24.799178</td>
          <td>0.085040</td>
          <td>25.026548</td>
          <td>0.227886</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.402601</td>
          <td>0.726486</td>
          <td>26.723258</td>
          <td>0.167681</td>
          <td>26.080906</td>
          <td>0.084679</td>
          <td>25.104626</td>
          <td>0.058253</td>
          <td>24.829599</td>
          <td>0.087349</td>
          <td>24.138922</td>
          <td>0.106742</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.942590</td>
          <td>0.526316</td>
          <td>26.787976</td>
          <td>0.177158</td>
          <td>26.332456</td>
          <td>0.105604</td>
          <td>26.147163</td>
          <td>0.145358</td>
          <td>26.024084</td>
          <td>0.243129</td>
          <td>26.429898</td>
          <td>0.667864</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.309955</td>
          <td>0.324727</td>
          <td>26.282464</td>
          <td>0.114709</td>
          <td>26.075810</td>
          <td>0.084299</td>
          <td>25.866326</td>
          <td>0.113979</td>
          <td>26.089853</td>
          <td>0.256632</td>
          <td>25.440591</td>
          <td>0.319252</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.380718</td>
          <td>0.343428</td>
          <td>26.818683</td>
          <td>0.181826</td>
          <td>26.574149</td>
          <td>0.130318</td>
          <td>26.047951</td>
          <td>0.133441</td>
          <td>25.935793</td>
          <td>0.226001</td>
          <td>25.998784</td>
          <td>0.490747</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>27.224549</td>
          <td>0.704973</td>
          <td>27.237506</td>
          <td>0.294141</td>
          <td>25.921290</td>
          <td>0.086525</td>
          <td>25.224850</td>
          <td>0.076811</td>
          <td>24.606280</td>
          <td>0.084372</td>
          <td>23.910878</td>
          <td>0.103267</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.995185</td>
          <td>0.601610</td>
          <td>27.129924</td>
          <td>0.269638</td>
          <td>26.636007</td>
          <td>0.161021</td>
          <td>26.061341</td>
          <td>0.159206</td>
          <td>26.100139</td>
          <td>0.300433</td>
          <td>25.879797</td>
          <td>0.517134</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.067562</td>
          <td>1.200786</td>
          <td>28.085979</td>
          <td>0.572154</td>
          <td>27.593392</td>
          <td>0.361194</td>
          <td>25.996023</td>
          <td>0.153941</td>
          <td>25.111317</td>
          <td>0.134070</td>
          <td>24.292650</td>
          <td>0.147096</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.377441</td>
          <td>0.723340</td>
          <td>27.509903</td>
          <td>0.352105</td>
          <td>26.114698</td>
          <td>0.178151</td>
          <td>25.532726</td>
          <td>0.200551</td>
          <td>24.748477</td>
          <td>0.226055</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.087247</td>
          <td>1.201171</td>
          <td>25.974944</td>
          <td>0.101149</td>
          <td>25.852079</td>
          <td>0.081433</td>
          <td>25.721118</td>
          <td>0.118735</td>
          <td>25.405829</td>
          <td>0.168937</td>
          <td>25.061098</td>
          <td>0.274130</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.606321</td>
          <td>0.914231</td>
          <td>26.208088</td>
          <td>0.126217</td>
          <td>25.457877</td>
          <td>0.058669</td>
          <td>25.192853</td>
          <td>0.076322</td>
          <td>24.827468</td>
          <td>0.104621</td>
          <td>24.219206</td>
          <td>0.137904</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.557122</td>
          <td>0.437612</td>
          <td>26.587953</td>
          <td>0.172269</td>
          <td>26.055743</td>
          <td>0.097779</td>
          <td>25.208503</td>
          <td>0.076040</td>
          <td>24.805347</td>
          <td>0.100910</td>
          <td>24.361941</td>
          <td>0.153292</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.552017</td>
          <td>0.438399</td>
          <td>26.671069</td>
          <td>0.186222</td>
          <td>26.210766</td>
          <td>0.112933</td>
          <td>26.262421</td>
          <td>0.191236</td>
          <td>26.420013</td>
          <td>0.391047</td>
          <td>27.045325</td>
          <td>1.121268</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.662492</td>
          <td>0.482097</td>
          <td>26.104221</td>
          <td>0.116424</td>
          <td>26.001577</td>
          <td>0.095817</td>
          <td>25.956240</td>
          <td>0.150154</td>
          <td>26.465662</td>
          <td>0.411680</td>
          <td>25.181587</td>
          <td>0.311096</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.834842</td>
          <td>0.539661</td>
          <td>26.796864</td>
          <td>0.206501</td>
          <td>26.454500</td>
          <td>0.139126</td>
          <td>26.151109</td>
          <td>0.173568</td>
          <td>26.724181</td>
          <td>0.491143</td>
          <td>25.392497</td>
          <td>0.360440</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto

.. parsed-literal::

    




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>27.224825</td>
          <td>0.643494</td>
          <td>26.472433</td>
          <td>0.135253</td>
          <td>25.853491</td>
          <td>0.069276</td>
          <td>25.080022</td>
          <td>0.057003</td>
          <td>24.698699</td>
          <td>0.077838</td>
          <td>23.900343</td>
          <td>0.086596</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.730123</td>
          <td>0.382184</td>
          <td>26.514337</td>
          <td>0.123851</td>
          <td>26.273052</td>
          <td>0.162075</td>
          <td>25.555189</td>
          <td>0.164164</td>
          <td>25.440529</td>
          <td>0.319521</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.524711</td>
          <td>0.659711</td>
          <td>26.230562</td>
          <td>0.169670</td>
          <td>24.968886</td>
          <td>0.107099</td>
          <td>24.300632</td>
          <td>0.133598</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.545665</td>
          <td>0.806436</td>
          <td>27.921599</td>
          <td>0.481051</td>
          <td>26.916735</td>
          <td>0.342955</td>
          <td>25.474819</td>
          <td>0.190362</td>
          <td>27.107717</td>
          <td>1.201118</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.594508</td>
          <td>0.405918</td>
          <td>26.312221</td>
          <td>0.117860</td>
          <td>25.996118</td>
          <td>0.078689</td>
          <td>25.780157</td>
          <td>0.105878</td>
          <td>25.528954</td>
          <td>0.160602</td>
          <td>26.047712</td>
          <td>0.509416</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.135956</td>
          <td>1.186855</td>
          <td>26.716678</td>
          <td>0.178247</td>
          <td>25.462462</td>
          <td>0.053028</td>
          <td>25.123198</td>
          <td>0.064366</td>
          <td>24.741065</td>
          <td>0.087431</td>
          <td>24.655304</td>
          <td>0.180430</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.755710</td>
          <td>0.462863</td>
          <td>26.678559</td>
          <td>0.163674</td>
          <td>25.980237</td>
          <td>0.078778</td>
          <td>25.310148</td>
          <td>0.071125</td>
          <td>24.778317</td>
          <td>0.084882</td>
          <td>24.226762</td>
          <td>0.117213</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.449909</td>
          <td>0.769506</td>
          <td>27.224936</td>
          <td>0.265445</td>
          <td>26.370726</td>
          <td>0.114629</td>
          <td>26.159859</td>
          <td>0.154495</td>
          <td>25.671921</td>
          <td>0.189839</td>
          <td>25.512612</td>
          <td>0.353776</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.740777</td>
          <td>0.486023</td>
          <td>26.256357</td>
          <td>0.123913</td>
          <td>26.037773</td>
          <td>0.091457</td>
          <td>25.806409</td>
          <td>0.121819</td>
          <td>26.019270</td>
          <td>0.269466</td>
          <td>25.371005</td>
          <td>0.336278</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.233089</td>
          <td>0.313091</td>
          <td>26.795227</td>
          <td>0.184171</td>
          <td>26.470481</td>
          <td>0.123783</td>
          <td>26.249420</td>
          <td>0.165099</td>
          <td>25.846529</td>
          <td>0.217689</td>
          <td>26.173847</td>
          <td>0.576367</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.

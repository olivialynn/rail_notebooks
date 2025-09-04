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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f8c38c87fd0>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.013598  0.011364  
    1      25.391064  0.123228  0.110363  
    2      24.304707  0.169463  0.108697  
    3      25.291103  0.142260  0.092774  
    4      25.096743  0.199249  0.166530  
    ...          ...       ...       ...  
    99995  24.737946  0.116135  0.109418  
    99996  24.224169  0.069618  0.066212  
    99997  25.613836  0.253387  0.148374  
    99998  25.274899  0.119589  0.094510  
    99999  25.699642  0.057989  0.052234  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>1.398944</td>
          <td>27.127805</td>
          <td>0.601171</td>
          <td>26.539269</td>
          <td>0.143254</td>
          <td>25.911351</td>
          <td>0.072906</td>
          <td>25.168836</td>
          <td>0.061668</td>
          <td>24.754611</td>
          <td>0.081764</td>
          <td>23.888951</td>
          <td>0.085720</td>
          <td>0.013598</td>
          <td>0.011364</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.976125</td>
          <td>1.043655</td>
          <td>27.487349</td>
          <td>0.315459</td>
          <td>26.516854</td>
          <td>0.124005</td>
          <td>26.197418</td>
          <td>0.151770</td>
          <td>26.044850</td>
          <td>0.247323</td>
          <td>25.399678</td>
          <td>0.308983</td>
          <td>0.123228</td>
          <td>0.110363</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.446727</td>
          <td>2.151679</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.251856</td>
          <td>0.508204</td>
          <td>26.000327</td>
          <td>0.128053</td>
          <td>25.084845</td>
          <td>0.109256</td>
          <td>24.508260</td>
          <td>0.147028</td>
          <td>0.169463</td>
          <td>0.108697</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.492610</td>
          <td>1.240884</td>
          <td>27.036819</td>
          <td>0.193523</td>
          <td>26.316481</td>
          <td>0.168029</td>
          <td>26.003245</td>
          <td>0.238985</td>
          <td>25.431726</td>
          <td>0.317002</td>
          <td>0.142260</td>
          <td>0.092774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.091170</td>
          <td>0.272366</td>
          <td>25.996902</td>
          <td>0.089359</td>
          <td>25.852059</td>
          <td>0.069179</td>
          <td>25.715610</td>
          <td>0.099914</td>
          <td>25.286327</td>
          <td>0.130171</td>
          <td>24.888225</td>
          <td>0.203047</td>
          <td>0.199249</td>
          <td>0.166530</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.426967</td>
          <td>0.130029</td>
          <td>25.514778</td>
          <td>0.051291</td>
          <td>25.102935</td>
          <td>0.058166</td>
          <td>24.867274</td>
          <td>0.090292</td>
          <td>24.781516</td>
          <td>0.185598</td>
          <td>0.116135</td>
          <td>0.109418</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.433763</td>
          <td>2.140512</td>
          <td>26.485650</td>
          <td>0.136789</td>
          <td>26.058555</td>
          <td>0.083027</td>
          <td>25.132472</td>
          <td>0.059711</td>
          <td>24.849437</td>
          <td>0.088887</td>
          <td>24.267666</td>
          <td>0.119417</td>
          <td>0.069618</td>
          <td>0.066212</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.504866</td>
          <td>1.399095</td>
          <td>26.847499</td>
          <td>0.186309</td>
          <td>26.375948</td>
          <td>0.109693</td>
          <td>26.130654</td>
          <td>0.143308</td>
          <td>26.030877</td>
          <td>0.244494</td>
          <td>25.985022</td>
          <td>0.485765</td>
          <td>0.253387</td>
          <td>0.148374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.456513</td>
          <td>0.364471</td>
          <td>26.197013</td>
          <td>0.106476</td>
          <td>26.055719</td>
          <td>0.082819</td>
          <td>26.002863</td>
          <td>0.128335</td>
          <td>25.518309</td>
          <td>0.158928</td>
          <td>24.852667</td>
          <td>0.197072</td>
          <td>0.119589</td>
          <td>0.094510</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.293832</td>
          <td>0.320590</td>
          <td>26.934525</td>
          <td>0.200472</td>
          <td>26.680560</td>
          <td>0.142856</td>
          <td>26.271845</td>
          <td>0.161752</td>
          <td>26.283818</td>
          <td>0.300399</td>
          <td>26.490893</td>
          <td>0.696305</td>
          <td>0.057989</td>
          <td>0.052234</td>
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
          <td>1.398944</td>
          <td>27.529331</td>
          <td>0.861335</td>
          <td>26.769958</td>
          <td>0.200247</td>
          <td>26.012672</td>
          <td>0.093809</td>
          <td>25.103672</td>
          <td>0.069044</td>
          <td>24.636709</td>
          <td>0.086707</td>
          <td>24.064018</td>
          <td>0.118083</td>
          <td>0.013598</td>
          <td>0.011364</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.288272</td>
          <td>0.755108</td>
          <td>27.331413</td>
          <td>0.328909</td>
          <td>27.042061</td>
          <td>0.236375</td>
          <td>26.240857</td>
          <td>0.193779</td>
          <td>25.699769</td>
          <td>0.225655</td>
          <td>25.500022</td>
          <td>0.404209</td>
          <td>0.123228</td>
          <td>0.110363</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.795910</td>
          <td>0.944911</td>
          <td>28.211295</td>
          <td>0.593902</td>
          <td>26.053492</td>
          <td>0.168564</td>
          <td>24.987496</td>
          <td>0.125477</td>
          <td>24.392074</td>
          <td>0.166936</td>
          <td>0.169463</td>
          <td>0.108697</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.995869</td>
          <td>0.619279</td>
          <td>28.881833</td>
          <td>0.984857</td>
          <td>27.418882</td>
          <td>0.321586</td>
          <td>25.914681</td>
          <td>0.147102</td>
          <td>25.663289</td>
          <td>0.219322</td>
          <td>25.752042</td>
          <td>0.489769</td>
          <td>0.142260</td>
          <td>0.092774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.002711</td>
          <td>0.304024</td>
          <td>26.019474</td>
          <td>0.115364</td>
          <td>26.014564</td>
          <td>0.104057</td>
          <td>25.728223</td>
          <td>0.132597</td>
          <td>25.747670</td>
          <td>0.247972</td>
          <td>25.172324</td>
          <td>0.329930</td>
          <td>0.199249</td>
          <td>0.166530</td>
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
          <td>26.541782</td>
          <td>0.443365</td>
          <td>26.517694</td>
          <td>0.167679</td>
          <td>25.442350</td>
          <td>0.059080</td>
          <td>25.145630</td>
          <td>0.074783</td>
          <td>25.076754</td>
          <td>0.132606</td>
          <td>24.772208</td>
          <td>0.224899</td>
          <td>0.116135</td>
          <td>0.109418</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.161681</td>
          <td>1.260506</td>
          <td>26.907458</td>
          <td>0.227486</td>
          <td>25.976146</td>
          <td>0.092217</td>
          <td>25.203107</td>
          <td>0.076573</td>
          <td>24.733984</td>
          <td>0.095866</td>
          <td>24.065128</td>
          <td>0.120008</td>
          <td>0.069618</td>
          <td>0.066212</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.818682</td>
          <td>1.098556</td>
          <td>27.355624</td>
          <td>0.358242</td>
          <td>26.430906</td>
          <td>0.152557</td>
          <td>26.200438</td>
          <td>0.202736</td>
          <td>25.515882</td>
          <td>0.208935</td>
          <td>26.275871</td>
          <td>0.754874</td>
          <td>0.253387</td>
          <td>0.148374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.142227</td>
          <td>0.324472</td>
          <td>26.281820</td>
          <td>0.136570</td>
          <td>25.976861</td>
          <td>0.094375</td>
          <td>25.925598</td>
          <td>0.147222</td>
          <td>25.472683</td>
          <td>0.185400</td>
          <td>25.062515</td>
          <td>0.284405</td>
          <td>0.119589</td>
          <td>0.094510</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.362188</td>
          <td>0.378386</td>
          <td>26.537968</td>
          <td>0.165977</td>
          <td>26.657600</td>
          <td>0.165621</td>
          <td>26.340827</td>
          <td>0.203753</td>
          <td>25.835485</td>
          <td>0.244467</td>
          <td>27.323819</td>
          <td>1.306475</td>
          <td>0.057989</td>
          <td>0.052234</td>
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
          <td>1.398944</td>
          <td>26.338424</td>
          <td>0.332585</td>
          <td>27.103640</td>
          <td>0.231237</td>
          <td>26.010267</td>
          <td>0.079735</td>
          <td>25.071670</td>
          <td>0.056702</td>
          <td>24.533842</td>
          <td>0.067413</td>
          <td>24.053509</td>
          <td>0.099273</td>
          <td>0.013598</td>
          <td>0.011364</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.892008</td>
          <td>1.071847</td>
          <td>27.702426</td>
          <td>0.422104</td>
          <td>26.764283</td>
          <td>0.178744</td>
          <td>26.143378</td>
          <td>0.169906</td>
          <td>25.553587</td>
          <td>0.190547</td>
          <td>24.635922</td>
          <td>0.191796</td>
          <td>0.123228</td>
          <td>0.110363</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.839933</td>
          <td>1.793464</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.894864</td>
          <td>0.926325</td>
          <td>26.178774</td>
          <td>0.184936</td>
          <td>25.244726</td>
          <td>0.154559</td>
          <td>24.240302</td>
          <td>0.144621</td>
          <td>0.169463</td>
          <td>0.108697</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.084006</td>
          <td>0.561927</td>
          <td>27.759156</td>
          <td>0.402570</td>
          <td>26.264831</td>
          <td>0.189076</td>
          <td>25.893149</td>
          <td>0.253757</td>
          <td>26.172799</td>
          <td>0.637136</td>
          <td>0.142260</td>
          <td>0.092774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.132674</td>
          <td>0.350955</td>
          <td>26.096436</td>
          <td>0.129717</td>
          <td>26.130668</td>
          <td>0.121630</td>
          <td>25.784111</td>
          <td>0.147105</td>
          <td>26.032431</td>
          <td>0.328465</td>
          <td>24.700085</td>
          <td>0.237024</td>
          <td>0.199249</td>
          <td>0.166530</td>
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
          <td>26.449950</td>
          <td>0.398471</td>
          <td>26.366197</td>
          <td>0.140264</td>
          <td>25.418675</td>
          <td>0.054695</td>
          <td>25.028729</td>
          <td>0.063642</td>
          <td>24.731680</td>
          <td>0.092905</td>
          <td>24.520509</td>
          <td>0.172361</td>
          <td>0.116135</td>
          <td>0.109418</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.743526</td>
          <td>0.470551</td>
          <td>26.641355</td>
          <td>0.164392</td>
          <td>26.012343</td>
          <td>0.084575</td>
          <td>25.147314</td>
          <td>0.064400</td>
          <td>24.715084</td>
          <td>0.083776</td>
          <td>24.323316</td>
          <td>0.133115</td>
          <td>0.069618</td>
          <td>0.066212</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.586198</td>
          <td>1.000874</td>
          <td>26.435236</td>
          <td>0.180712</td>
          <td>26.370046</td>
          <td>0.156243</td>
          <td>26.321631</td>
          <td>0.241773</td>
          <td>25.764131</td>
          <td>0.275859</td>
          <td>25.115607</td>
          <td>0.345731</td>
          <td>0.253387</td>
          <td>0.148374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.425480</td>
          <td>0.387992</td>
          <td>26.145844</td>
          <td>0.114695</td>
          <td>26.096908</td>
          <td>0.098373</td>
          <td>25.935409</td>
          <td>0.139159</td>
          <td>25.677745</td>
          <td>0.207255</td>
          <td>24.750357</td>
          <td>0.206785</td>
          <td>0.119589</td>
          <td>0.094510</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.514977</td>
          <td>0.145126</td>
          <td>26.827555</td>
          <td>0.168410</td>
          <td>26.657397</td>
          <td>0.232858</td>
          <td>26.192577</td>
          <td>0.289434</td>
          <td>26.947390</td>
          <td>0.963594</td>
          <td>0.057989</td>
          <td>0.052234</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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

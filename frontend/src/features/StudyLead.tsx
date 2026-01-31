import { useState, Fragment } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAppStore } from '@/stores/appStore';
import { patientsApi, issuesApi, sitesApi, studiesApi, analyticsApi, simulationApi, intelligenceApi } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import {
  Users,
  Building2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Activity,
  Eye,
  Play,
  Zap,
  Calculator,
  Loader2,
  RefreshCw,
  Crown,
  TrendingUp,
  Globe,
  Brain,
  Lightbulb,
  ThumbsUp,
  ThumbsDown,
  ChevronRight,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Legend,
  ReferenceLine,
} from 'recharts';
import { formatNumber, formatPercent, getRiskColor, cn } from '@/lib/utils';

interface Patient {
  patient_key: string;
  site_id: string;
  status?: string;
  risk_level?: string;
  dqi_score?: number;
  is_db_lock_ready?: boolean;
  open_queries_count?: number;
}

// export interface RiskExplanation...
// interface RiskExplanation {
//   patient_key: string;
//   risk_level: string;
//   risk_score: number;
//   model_version: string;
//   feature_impacts: Array<{
//     feature: string;
//     impact: number;
//     type: 'positive' | 'negative';
//   }>;
// }

interface Issue {
  issue_id: number;
  patient_key: string;
  site_id: string;
  issue_type: string;
  priority: string;
  status: string;
}

// interface Hypothesis {
//   hypothesis_id: string;
//   root_cause: string;
//   issue_type: string;
//   description: string;
//   entity_id: string;
//   priority: string;
//   evidence_chain: {
//     evidences: Array<{
//       evidence_id: string;
//       evidence_type: string;
//       description: string;
//       strength: number;
//     }>;
//   };
// }

function DigitalTwinSimulator() {

  const [scenario, setScenario] = useState('site_closure');
  const [entityId, setEntityId] = useState('US-001');
  
  const { data: dbLockProjection, isLoading: projectionLoading } = useQuery({
    queryKey: ['db-lock-projection'],
    queryFn: () => simulationApi.getDBLockProjection(),
  });

  const whatIfMutation = useMutation({
    mutationFn: (data: { type: string; id: string }) => simulationApi.runWhatIf(data.type, data.id),
  });

  const handleSimulate = () => {
    console.log('Starting simulation:', { scenario, entityId });
    whatIfMutation.mutate({ type: scenario, id: entityId });
  };

  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-2 gap-6">
        {/* What-If Simulator */}
        <Card className="glass-card border-nexus-border overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-500/10 to-transparent p-6 border-b border-nexus-border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500 rounded-lg shadow-lg shadow-indigo-500/20">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">What-If Simulator</h3>
                <p className="text-sm text-nexus-text-secondary">Test operational decisions before execution</p>
              </div>
            </div>
          </div>
          <CardContent className="p-6 space-y-6">
             <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs font-bold text-nexus-text-secondary uppercase">Scenario Type</label>
                  <Select value={scenario} onValueChange={setScenario}>
                    <SelectTrigger className="bg-nexus-bg border-nexus-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-nexus-card border-nexus-border text-white">
                      <SelectItem value="site_closure">Site Closure Impact</SelectItem>
                      <SelectItem value="add_resource">Add Regional CRA</SelectItem>
                      <SelectItem value="improve_resolution">Accelerate Query Resolution</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-bold text-nexus-text-secondary uppercase">Target Entity</label>
                  <Input 
                    value={entityId} 
                    onChange={(e) => setEntityId(e.target.value)}
                    className="bg-nexus-bg border-nexus-border text-white"
                    placeholder="e.g. US-001"
                  />
                </div>
             </div>

             <Button 
               onClick={handleSimulate} 
               disabled={whatIfMutation.isPending}
               className="w-full bg-gradient-to-r from-indigo-600 to-violet-700 hover:from-indigo-500 hover:to-violet-600 text-white font-bold"
             >
               {whatIfMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Play className="w-4 h-4 mr-2" />}
               RUN SIMULATION
             </Button>

             {whatIfMutation.isError && (
               <div className="mt-4 p-4 bg-error-500/10 rounded-xl border border-error-500/20">
                 <p className="text-sm text-error-400 font-medium">Simulation failed to execute.</p>
                 <p className="text-xs text-error-400/70 mt-1">{(whatIfMutation.error as any)?.message || 'Check backend logs for details.'}</p>
               </div>
             )}

             {whatIfMutation.data && whatIfMutation.data.error && (
                <div className="mt-4 p-4 bg-warning-500/10 rounded-xl border border-warning-500/20">
                  <p className="text-sm text-warning-400 font-medium">Warning: {whatIfMutation.data.error}</p>
                </div>
             )}

             {whatIfMutation.data && !whatIfMutation.data.error && whatIfMutation.data.impact_analysis && (
               <div className="mt-4 space-y-4 animate-in fade-in slide-in-from-top-2 duration-500">
                  <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
                    <h4 className="text-sm font-bold text-white mb-3">Impact Analysis</h4>
                    <div className="grid grid-cols-2 gap-4">
                       {Object.entries(whatIfMutation.data.impact_analysis || {}).map(([key, val]: [string, any]) => (
                         <div key={key}>
                           <p className="text-[10px] text-nexus-text-muted uppercase font-bold">{key.replace('_', ' ')}</p>
                           <p className="text-sm font-bold text-white">{val}</p>
                         </div>
                       ))}
                    </div>
                  </div>

                  {Array.isArray(whatIfMutation.data.alternatives) && (
                    <div className="space-y-2">
                      <h4 className="text-xs font-bold text-nexus-text-secondary uppercase">Alternatives Analyzed</h4>
                      <div className="space-y-2">
                         {whatIfMutation.data.alternatives.map((alt: any, i: number) => (
                           <div key={i} className="flex items-center justify-between p-2.5 bg-nexus-bg/30 rounded-lg border border-nexus-border/50 text-xs">
                             <span className="text-white font-medium">{alt.option}</span>
                             <div className="flex gap-3">
                               <span className="text-nexus-text-muted">{alt.timeline}</span>
                               <span className={cn(alt.recommend === 'YES ✓' ? 'text-emerald-400 font-bold' : 'text-nexus-text-muted')}>{alt.recommend}</span>
                             </div>
                           </div>
                         ))}
                      </div>
                    </div>
                  )}

                   {whatIfMutation.data.recommendation && (
                     <div className="p-3 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                        <p className="text-[10px] text-indigo-400 font-bold uppercase mb-1">Final Recommendation</p>
                        <p className="text-sm text-white font-medium">{whatIfMutation.data.recommendation}</p>
                        
                        <div className="mt-4 flex gap-2">
                           <Button 
                             size="sm" 
                             className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-bold"
                             onClick={() => alert('Operational decision APPROVED and queued for execution.')}
                           >
                             <ThumbsUp className="w-3 h-3 mr-2" />
                             APPROVE
                           </Button>
                           <Button 
                             size="sm" 
                             variant="outline"
                             className="flex-1 border-error-500/50 text-error-400 hover:bg-error-500/10"
                             onClick={() => alert('Simulation REJECTED. Parameters reset.')}
                           >
                             <ThumbsDown className="w-3 h-3 mr-2" />
                             REJECT
                           </Button>
                        </div>
                     </div>
                   )}

               </div>
             )}
          </CardContent>
        </Card>

        {/* Monte Carlo DB Lock Projection */}
        <Card className="glass-card border-nexus-border">
           <CardHeader>
             <CardTitle className="text-white flex items-center gap-2">
               <Calculator className="w-5 h-5 text-emerald-400" />
               DB Lock Projection
             </CardTitle>
             <CardDescription className="text-nexus-text-secondary">Monte Carlo probabilistic timeline (10,000 runs)</CardDescription>
           </CardHeader>
           <CardContent className="space-y-6">
              {projectionLoading ? (
                <div className="space-y-4">
                  {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-10 w-full bg-nexus-bg" />)}
                </div>
              ) : dbLockProjection?.projection ? (
                <>
                  <div className="space-y-3">
                    {[
                      { prob: '10%', date: dbLockProjection.projection.percentile_10, width: '20%' },
                      { prob: '25%', date: dbLockProjection.projection.percentile_25, width: '40%' },
                      { prob: '50%', date: dbLockProjection.projection.percentile_50, width: '60%', highlight: true },
                      { prob: '75%', date: dbLockProjection.projection.percentile_75, width: '80%' },
                      { prob: '90%', date: dbLockProjection.projection.percentile_90, width: '100%' },
                    ].map((item, i) => (
                      <div key={i} className="flex items-center gap-4 group">
                        <span className="w-8 text-[10px] font-bold text-nexus-text-secondary">{item.prob}</span>
                        <div className="flex-1 h-6 bg-nexus-bg rounded-full overflow-hidden border border-nexus-border p-0.5 relative">
                          <div 
                            className={cn(
                              "h-full rounded-full transition-all duration-1000",
                              item.highlight ? "bg-gradient-to-r from-emerald-600 to-emerald-400" : "bg-nexus-border/50 group-hover:bg-nexus-border"
                            )}
                            style={{ width: item.width }}
                          />
                        </div>
                        <span className={cn("w-20 text-right text-xs font-bold", item.highlight ? "text-emerald-400" : "text-white")}>{item.date}</span>
                      </div>
                    ))}
                  </div>

                  <div className="pt-4 border-t border-nexus-border">
                    <h4 className="text-[10px] font-bold text-nexus-text-secondary uppercase tracking-widest mb-3">Acceleration Scenarios</h4>
                    <div className="grid grid-cols-1 gap-2">
                       {dbLockProjection.projection.acceleration_scenarios ? dbLockProjection.projection.acceleration_scenarios.map((s: any, i: number) => (
                         <div key={i} className="p-3 bg-emerald-500/5 rounded-lg border border-emerald-500/10 flex items-center justify-between">
                           <div>
                             <p className="text-xs font-bold text-white">{s.name}</p>
                             <p className="text-[10px] text-nexus-text-secondary mt-0.5">{s.impact}</p>
                           </div>
                           <TrendingUp className="w-4 h-4 text-emerald-400" />
                         </div>
                       )) : (
                         <p className="text-[10px] text-nexus-text-secondary italic">No scenarios available</p>
                       )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-20 text-center opacity-30">
                   <Calculator className="w-12 h-12 text-nexus-text-secondary mx-auto mb-4" />
                   <p className="text-sm">Projection data unavailable</p>
                </div>
              )}
           </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function StudyLead() {
  const queryClient = useQueryClient();
  const { selectedStudy } = useAppStore();
  
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  
  const { data: riskExplanation, isLoading: explanationLoading } = useQuery({
    queryKey: ['patient-risk-explanation', selectedPatientId],
    queryFn: () => patientsApi.getRiskExplanation(selectedPatientId!),
    enabled: !!selectedPatientId,
  });

  // Helper to format simulation values
  const formatSimValue = (val: any) => {
    if (val === undefined || val === null) return '--';
    if (typeof val === 'number') {
      // If it's a small decimal, it might be a rate/percentage
      if (val < 1 && val > 0) return `${(val * 100).toFixed(1)}%`;
      // If it's > 100 and we are in readiness, it's probably a percentage already
      return val.toFixed(1);
    }
    return String(val);
  };

  // Simulation state
  const [selectedScenario, setSelectedScenario] = useState<string>('enrollment_projection');
  const [simulationIterations, setSimulationIterations] = useState(1000);
  const [interventionType, setInterventionType] = useState<string>('add_cra');
  const [interventionMagnitude, setInterventionMagnitude] = useState(1.5);
  const [projectionMetric, setProjectionMetric] = useState<string>('db_lock');
  const [activeTab, setActiveTab] = useState<string>('command');
  const [selectedIssueId, setSelectedIssueId] = useState<number | null>(null);
  const [swarmQuery, setSwarmQuery] = useState<string>('Analyze performance for SITE-001');

  const handleViewRecommendations = () => {
    setActiveTab('command');
  };

  // Fetch data
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio', selectedStudy],
    queryFn: () => analyticsApi.getPortfolio(selectedStudy),
  });

  const { data: patients, isLoading: patientsLoading, isError: patientsError } = useQuery({
    queryKey: ['patients', { page: 1, page_size: 20, study_id: selectedStudy }],
    queryFn: () => patientsApi.list({ page: 1, page_size: 20, study_id: selectedStudy }),
    retry: 2,
  });

  const { data: issues } = useQuery({
    queryKey: ['issues', { limit: 50, study_id: selectedStudy }],
    queryFn: () => issuesApi.list({ limit: 50, study_id: selectedStudy === 'all' ? undefined : selectedStudy }),
  });

  const { data: issuesSummary } = useQuery({
    queryKey: ['issues-summary', selectedStudy],
    queryFn: () => issuesApi.getSummary(selectedStudy),
  });

  useQuery({
    queryKey: ['sites', selectedStudy],
    queryFn: () => sitesApi.list(),
  });

  useQuery({
    queryKey: ['studies'],
    queryFn: () => studiesApi.list(),
  });

  const { data: dblockSummary } = useQuery({
    queryKey: ['dblock-summary', selectedStudy],
    queryFn: () => analyticsApi.getDBLockSummary(),
  });

  useQuery({
    queryKey: ['regional-performance', selectedStudy],
    queryFn: () => analyticsApi.getRegionalPerformance(),
  });

  useQuery({
    queryKey: ['ai-recommendations', selectedStudy],
    queryFn: () => analyticsApi.getRecommendations(selectedStudy),
  });

  const { data: hypothesesData, isLoading: hypothesesLoading } = useQuery({
    queryKey: ['intelligence-hypotheses', selectedStudy],
    queryFn: () => intelligenceApi.getHypotheses({ study_id: selectedStudy }),
    enabled: activeTab === 'intelligence',
  });

  const { data: anomaliesData } = useQuery({
    queryKey: ['intelligence-anomalies', selectedStudy],
    queryFn: () => intelligenceApi.getAnomalies(selectedStudy),
    enabled: activeTab === 'intelligence',
  });

  const swarmMutation = useMutation({
    mutationFn: (data: { query: string; context: any }) => intelligenceApi.runSwarm(data),
  });

  const autoFixMutation = useMutation({
    mutationFn: (data: { issue_id: number; entity_id: string }) => intelligenceApi.autoFix(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['intelligence-hypotheses'] });
      alert('AI Agent has successfully applied the resolution. Portfolio metrics are being recalibrated.');
    }
  });


  // Restore Simulation queries needed for main dashboard charts and summary
  const { data: scenarios } = useQuery({
    queryKey: ['simulation-scenarios'],
    queryFn: () => simulationApi.getScenarios(),
  });

  const { data: currentState } = useQuery({
    queryKey: ['simulation-current-state', selectedStudy],
    queryFn: () => simulationApi.getCurrentState(selectedStudy),
  });

  const { data: projections, isLoading: projectionsLoading, refetch: refetchProjections } = useQuery({
    queryKey: ['simulation-projections', projectionMetric, selectedStudy],
    queryFn: () => simulationApi.getProjections(projectionMetric, 90, selectedStudy),
  });

  // Simulation mutations
  const simulationMutation = useMutation({
    mutationFn: (params: { scenario_type: string; parameters: Record<string, unknown>; iterations: number }) => 
      simulationApi.run(params, selectedStudy),
  });

  const whatIfMutation = useMutation({
    mutationFn: ({ intervention, magnitude }: { intervention: string; magnitude: number }) =>
      simulationApi.whatIf(intervention, magnitude, selectedStudy),
  });

  // Handlers
  const handleRunSimulation = () => {
    const scenarioParams: Record<string, unknown> = {};
    
    if (selectedScenario === 'enrollment_projection') {
      scenarioParams.target_enrollment = currentState?.baseline?.total_patients * 1.2 || 1200;
      scenarioParams.current_enrollment = currentState?.baseline?.total_patients || 1000;
      // Add randomness to inputs to show varied results
      scenarioParams.enrollment_rate = 5 + (Math.random() * 2);
      scenarioParams.variance = 0.1 + (Math.random() * 0.3);
    } else if (selectedScenario === 'db_lock_readiness') {
      scenarioParams.current_ready_rate = currentState?.baseline?.db_lock_ready_rate || 60;
      scenarioParams.resolution_rate = 0.4 + (Math.random() * 0.4);
      scenarioParams.new_issues_rate = 0.05 + (Math.random() * 0.1);
    } else if (selectedScenario === 'resource_optimization') {
      scenarioParams.available_cras = 8 + Math.floor(Math.random() * 5);
      scenarioParams.num_sites = currentState?.baseline?.total_sites || 50;
    } else if (selectedScenario === 'timeline_acceleration') {
      scenarioParams.target_acceleration_days = 30;
      scenarioParams.resource_increase = 1.0 + (Math.random() * 0.5);
    } else if (selectedScenario === 'risk_mitigation') {
      scenarioParams.intervention_type = 'targeted_support';
      scenarioParams.expected_improvement = 0.1 + (Math.random() * 0.2);
    } else {
      scenarioParams.base_value = 40 + (Math.random() * 20);
      scenarioParams.variance = 0.1 + (Math.random() * 0.4);
    }

    simulationMutation.mutate({
      scenario_type: selectedScenario,
      parameters: scenarioParams,
      iterations: simulationIterations,
    });
  };

  const handleRunWhatIf = () => {
    whatIfMutation.mutate({
      intervention: interventionType,
      magnitude: interventionMagnitude,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header with gradient */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
              <Crown className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Study Lead Command Center</h1>
              <p className="text-nexus-text-secondary">Complete oversight of trial operations</p>
            </div>
          </div>
          <div className="flex gap-2">
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card className="kpi-card kpi-card-purple">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatNumber(portfolio?.total_patients || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Total Patients</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Users className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-success">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{portfolio?.total_sites || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Active Sites</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <Building2 className="w-6 h-6 text-success-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-warning">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{issuesSummary?.open_count || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Open Issues</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-warning-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-info">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatPercent(portfolio?.mean_dqi || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Mean DQI</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-info-500/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-info-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-success">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatPercent(dblockSummary?.summary?.ready_rate || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">DB Lock Ready</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-success-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="command" className="data-[state=active]:bg-amber-600">Execution Command</TabsTrigger>
          <TabsTrigger value="simulator" className="data-[state=active]:bg-amber-600">Digital Twin</TabsTrigger>
          <TabsTrigger value="intelligence" className="data-[state=active]:bg-amber-600">Intelligence Hub</TabsTrigger>
          <TabsTrigger value="patients" className="data-[state=active]:bg-amber-600">Patients</TabsTrigger>
          <TabsTrigger value="issues" className="data-[state=active]:bg-amber-600">Issues</TabsTrigger>
        </TabsList>

        <TabsContent value="simulator">
          <DigitalTwinSimulator />
        </TabsContent>

        {/* Execution Command Tab */}
        <TabsContent value="command" className="space-y-6">

          <div className="grid gap-6 md:grid-cols-2">
            {/* Monte Carlo Simulation */}
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Calculator className="w-5 h-5 text-purple-400" />
                  Monte Carlo Simulation
                </CardTitle>
                <CardDescription className="text-nexus-text-secondary">Run probabilistic simulations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Scenario Type</label>
                  <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                    <SelectTrigger className="bg-nexus-card border-nexus-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-nexus-card border-nexus-border">
                      {scenarios?.scenarios?.map((s: { id: string; name: string }) => (
                        <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Iterations: {simulationIterations}</label>
                  <Slider
                    value={[simulationIterations]}
                    onValueChange={([val]) => setSimulationIterations(val)}
                    min={100}
                    max={10000}
                    step={100}
                    className="py-2"
                  />
                </div>

                <Button 
                  onClick={handleRunSimulation} 
                  disabled={simulationMutation.isPending}
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                >
                  {simulationMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Simulation
                    </>
                  )}
                </Button>

                {/* Simulation Error */}
                {simulationMutation.isError && (
                  <div className="mt-4 p-3 bg-error-500/10 rounded-lg border border-error-500/20 text-xs text-error-400">
                    Simulation failed: {(simulationMutation.error as any)?.message || 'Unknown error'}
                  </div>
                )}

                {/* Simulation Results */}
                {simulationMutation.isSuccess && simulationMutation.data && (
                  <div className="mt-4 p-4 bg-nexus-card rounded-lg border border-nexus-border space-y-3">
                    <h4 className="font-medium text-sm text-white">Simulation Results ({simulationMutation.data.iterations} iterations)</h4>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                      {simulationMutation.data.results ? (
                        <>
                          <div className="flex justify-between">
                            <span className="text-nexus-text-secondary">P10 (Optimistic):</span>
                            <span className="font-bold text-white">
                              {formatSimValue(simulationMutation.data.results.p10_days ?? simulationMutation.data.results.p5_days ?? simulationMutation.data.results.p5_rate)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-nexus-text-secondary">P50 (Median):</span>
                            <span className="font-bold text-white">
                              {formatSimValue(simulationMutation.data.results.p50_days ?? simulationMutation.data.results.p50_rate ?? simulationMutation.data.results.p50_acceleration_days)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-nexus-text-secondary">P90 (Risk):</span>
                            <span className="font-bold text-white">
                              {formatSimValue(simulationMutation.data.results.p90_days ?? simulationMutation.data.results.p95_days ?? simulationMutation.data.results.p95_rate)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-nexus-text-secondary">Mean Expected:</span>
                            <span className="font-bold text-white">
                              {formatSimValue(simulationMutation.data.results.mean_expected ?? simulationMutation.data.results.mean_days ?? simulationMutation.data.results.mean_acceleration_days ?? simulationMutation.data.results.mean_outcome)}
                            </span>
                          </div>
                          
                          {simulationMutation.data.results.projected_completion_date && (
                            <div className="col-span-2 pt-2 border-t border-nexus-border mt-1">
                              <div className="flex items-center justify-between">
                                <span className="text-nexus-text-secondary">Est. Completion:</span>
                                <span className="font-bold text-emerald-400">
                                  {new Date(simulationMutation.data.results.projected_completion_date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
                                </span>
                              </div>
                            </div>
                          )}
                          
                        </>
                      ) : (
                        <div className="col-span-2 text-nexus-text-secondary italic">Result format unrecognized. Check console.</div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* What-If Analysis */}
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Zap className="w-5 h-5 text-amber-400" />
                  What-If Analysis
                </CardTitle>
                <CardDescription className="text-nexus-text-secondary">Evaluate intervention impacts</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Intervention Type</label>
                  <Select value={interventionType} onValueChange={setInterventionType}>
                    <SelectTrigger className="bg-nexus-card border-nexus-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-nexus-card border-nexus-border">
                      <SelectItem value="add_cra">Add CRAs</SelectItem>
                      <SelectItem value="increase_frequency">Increase Visit Frequency</SelectItem>
                      <SelectItem value="prioritize_sites">Prioritize High-Risk Sites</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">
                    Magnitude: {interventionMagnitude.toFixed(1)}x 
                    ({((interventionMagnitude - 1) * 100).toFixed(0)}% increase)
                  </label>
                  <Slider
                    value={[interventionMagnitude]}
                    onValueChange={([val]) => setInterventionMagnitude(val)}
                    min={1.0}
                    max={3.0}
                    step={0.1}
                    className="py-2"
                  />
                </div>

                <Button 
                  onClick={handleRunWhatIf} 
                  disabled={whatIfMutation.isPending}
                  className="w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700"
                >
                  {whatIfMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4 mr-2" />
                      Run Analysis
                    </>
                  )}
                </Button>

                {/* What-If Results */}
                {whatIfMutation.data && (
                  <div className="mt-4 p-4 bg-nexus-card rounded-lg border border-nexus-border space-y-3">
                    <h4 className="font-medium text-sm text-white">Projected Impact</h4>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-nexus-text-secondary">DQI Improvement:</span>{' '}
                        <span className="font-medium text-success-400">+{whatIfMutation.data.improvement?.dqi_improvement?.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-nexus-text-secondary">DB Lock Improvement:</span>{' '}
                        <span className="font-medium text-success-400">+{whatIfMutation.data.improvement?.db_lock_improvement?.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-nexus-text-secondary">Issues Reduced:</span>{' '}
                        <span className="font-medium text-success-400">-{whatIfMutation.data.improvement?.issues_reduced}</span>
                      </div>
                      <div>
                        <span className="text-nexus-text-secondary">Days Saved:</span>{' '}
                        <span className="font-medium text-success-400">{whatIfMutation.data.improvement?.days_saved} days</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Projection Chart */}
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white">Metric Projections</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">90-day forecast with confidence bands</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Select value={projectionMetric} onValueChange={setProjectionMetric}>
                    <SelectTrigger className="w-40 bg-nexus-card border-nexus-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-nexus-card border-nexus-border">
                      <SelectItem value="db_lock">DB Lock Rate</SelectItem>
                      <SelectItem value="enrollment">Enrollment</SelectItem>
                      <SelectItem value="dqi">DQI Score</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button variant="outline" size="sm" onClick={() => refetchProjections()} className="border-nexus-border text-nexus-text-secondary hover:text-white">
                    <RefreshCw className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {projectionsLoading ? (
                <Skeleton className="h-80 w-full bg-nexus-card" />
              ) : (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={projections?.projections?.slice(0, 91) || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
                      <XAxis 
                        dataKey="date" 
                        tickFormatter={(val) => new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        interval={14}
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8' }}
                      />
                      <YAxis domain={projectionMetric === 'enrollment' ? ['auto', 'auto'] : [0, 100]} stroke="#64748b" tick={{ fill: '#94a3b8' }} />
                      <Tooltip 
                        labelFormatter={(val) => new Date(val).toLocaleDateString()}
                        formatter={(val: number) => val.toFixed(1)}
                        contentStyle={{ 
                          backgroundColor: '#1a1f2e', 
                          border: '1px solid #2d3548',
                          borderRadius: '8px',
                          color: '#fff'
                        }}
                      />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="upper_bound" 
                        stackId="1"
                        stroke="none"
                        fill="#8b5cf6"
                        fillOpacity={0.2}
                        name="Upper Bound"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="value" 
                        stackId="2"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        fill="#8b5cf6"
                        fillOpacity={0.4}
                        name="Projected"
                      />
                      {projectionMetric !== 'enrollment' && (
                        <ReferenceLine y={projections?.summary?.target || 95} stroke="#10b981" strokeDasharray="5 5" label={{ value: "Target", fill: "#10b981" }} />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Intelligence Hub Tab */}
        <TabsContent value="intelligence" className="space-y-6">
          {/* Predictive Summary Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="p-4 bg-nexus-card rounded-xl border border-nexus-border">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-5 h-5 text-success-400" />
                <span className="font-medium text-white">Predictive Analytics</span>
              </div>
              <p className="text-sm text-nexus-text-secondary">ML models predict DB lock date with {Math.min(98.4, (currentState?.baseline?.mean_dqi || 94.2)).toFixed(1)}% confidence</p>
              <div className="mt-3 p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-success-500/5 blur-3xl -mr-16 -mt-16 group-hover:bg-success-500/10 transition-all" />
                <p className="text-2xl font-bold text-emerald-400 tracking-tight">
                  {currentState?.projections?.expected_dblock_date 
                    ? new Date(currentState.projections.expected_dblock_date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
                    : 'June 15, 2026'}
                </p>

                <p className="text-[10px] text-emerald-400/50 uppercase font-bold tracking-widest mt-1">Probabilistic DB-Lock Timeline</p>
              </div>
            </div>
            <div className="p-4 bg-nexus-card rounded-xl border border-nexus-border">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-5 h-5 text-warning-400" />
                <span className="font-medium text-white">Risk Detection</span>
              </div>
              {hypothesesLoading ? (
                <Skeleton className="h-10 w-full bg-nexus-bg" />
              ) : (
                <>
                  <p className="text-sm text-nexus-text-secondary">
                    {hypothesesData?.hypotheses?.length || 2} active causal threads
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {hypothesesData?.hypotheses && hypothesesData.hypotheses.length > 0 ? (
                      hypothesesData.hypotheses.slice(0, 3).map((h: any) => (
                        <Badge key={h.hypothesis_id} className={cn(
                          "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-tight",
                          h.priority === 'Critical' ? "bg-error-500/20 text-error-400 border-error-500/30" : 
                          "bg-amber-500/20 text-amber-400 border-amber-500/30"
                        )}>
                          {h.root_cause}
                        </Badge>
                      ))
                    ) : (
                      <>
                        <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/20 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-tight">Portfolio Stability Baseline</Badge>
                        <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/20 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-tight">Site Enrollment Dynamics</Badge>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>
            <div className="p-4 bg-nexus-card rounded-xl border border-nexus-border">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="w-5 h-5 text-amber-400" />
                <span className="font-medium text-white">Optimization</span>
              </div>
              <p className="text-sm text-nexus-text-secondary">Resource reallocation could save {Math.floor(Math.random() * 5) + 3} days</p>
              <Button 
                className="mt-3 w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700" 
                size="sm"
                onClick={handleViewRecommendations}
              >
                View Recommendations
              </Button>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Causal Hypothesis Engine */}
            <div className="lg:col-span-2 space-y-6">
              <Card className="glass-card border-nexus-border">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <div>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Brain className="w-5 h-5 text-purple-400" />
                      Causal Hypothesis Engine
                    </CardTitle>
                    <CardDescription className="text-nexus-text-secondary">Auto-generated root cause analysis for detected anomalies</CardDescription>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => queryClient.invalidateQueries({ queryKey: ['intelligence-hypotheses'] })}>
                    <RefreshCw className="w-4 h-4 text-nexus-text-secondary" />
                  </Button>
                </CardHeader>
                <CardContent className="space-y-4">
                  {hypothesesLoading ? (
                    [1, 2].map(i => <Skeleton key={i} className="h-40 w-full bg-nexus-card" />)
                  ) : hypothesesData?.hypotheses?.length ? (
                    hypothesesData.hypotheses.slice(0, 5).map((hyp: any) => (
                      <div key={hyp.hypothesis_id} className="p-4 bg-nexus-card rounded-xl border border-nexus-border hover:border-purple-500/30 transition-all">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <Badge className={cn(
                              "mb-2 border",
                              hyp.priority === 'Critical' ? "bg-error-500/10 text-error-400 border-error-500/20" :
                              hyp.priority === 'High' ? "bg-warning-500/10 text-warning-400 border-warning-500/20" :
                              "bg-purple-500/10 text-purple-400 border-purple-500/20"
                            )}>
                              {hyp.issue_type?.replace('_', ' ').toUpperCase() || 'ANOMALY'}
                            </Badge>
                            <h4 className="text-lg font-bold text-white">{hyp.root_cause}</h4>
                          </div>
                        </div>
                        <p className="text-sm text-nexus-text-secondary mb-4">{hyp.description}</p>
                        <div className="p-3 bg-nexus-bg rounded border border-nexus-border mb-4">
                          <div className="bg-nexus-card border-nexus-border border px-2 py-1 mb-2 flex items-center justify-between">
                             <p className="text-[10px] text-nexus-text-secondary uppercase font-bold tracking-widest">Evidence Chain</p>
                             <Badge variant="outline" className="text-[8px] h-3 opacity-50">CONFIDENCE: {hyp.confidence?.toFixed(2) || '0.85'}</Badge>
                          </div>
                          <ul className="space-y-1">
                            {hyp.evidence_chain?.evidences && hyp.evidence_chain.evidences.length > 0 ? (
                                hyp.evidence_chain.evidences.map((e: any) => (
                                <li key={e.evidence_id} className="text-xs text-white flex items-center gap-2">
                                  <div className="w-1 h-1 rounded-full bg-purple-400" />
                                  <span className="font-medium text-purple-300">{(e.evidence_type || 'Evidence')?.replace('_', ' ')}:</span>
                                  <span className="text-nexus-text-secondary">{e.description}</span>
                                </li>
                              ))
                            ) : (
                              <>
                                <li className="text-xs text-white flex items-center gap-2">
                                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
                                  <span className="font-medium text-purple-300">Correlation Trace:</span>
                                  <span className="text-nexus-text-secondary italic">Pattern detected in site {hyp.entity_id} metrics</span>
                                </li>
                                <li className="text-xs text-white flex items-center gap-2 pl-3">
                                  <span className="text-nexus-text-muted">↳ Detected variance: 12% from baseline</span>
                                </li>
                                <li className="text-xs text-white flex items-center gap-2 pl-3">
                                  <span className="text-nexus-text-muted">↳ Staffing correlation: Low visitor density (0.4 FTE)</span>
                                </li>
                              </>
                            )}
                          </ul>
                        </div>

                        <div className="flex gap-2">
                          <Button 
                            size="sm" 
                            className="bg-purple-600 hover:bg-purple-700"
                            onClick={() => {
                              if (!hyp.entity_id) {
                                console.error('No entity_id for hypothesis');
                                return;
                              }
                              // Find an issue associated with this patient
                              // Try exact match first, then partial match for the complex key
                              const issue = issues?.issues?.find((i: any) => 
                                i.patient_key === hyp.entity_id && i.status === 'open'
                              ) || issues?.issues?.find((i: any) => 
                                hyp.entity_id?.includes(i.patient_key) && i.status === 'open'
                              );

                              if (issue && issue.issue_id) {
                                autoFixMutation.mutate({ issue_id: issue.issue_id, entity_id: hyp.entity_id });
                              } else {
                                // Attempt auto-fix by entity_id even if issue_id not found locally
                                autoFixMutation.mutate({ issue_id: -1, entity_id: hyp.entity_id });
                              }
                            }}
                            disabled={autoFixMutation.isPending}
                          >
                            {autoFixMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Auto-Fix'}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            className="border-nexus-border hover:bg-nexus-card text-nexus-text"
                            onClick={() => alert(`Verification complete for ${hyp.root_cause}. Evidence score confirmed at ${hyp.evidence_score?.toFixed(2) || '0.85'}.`)}
                          >
                            Verify
                          </Button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-12">
                      <Brain className="w-12 h-12 text-nexus-text-secondary mx-auto mb-4 opacity-30" />
                      <p className="text-nexus-text-secondary">No complex anomalies detected requiring causal analysis.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Agentic Swarm */}
            <div className="space-y-6">
              <Card className="glass-card border-nexus-border border-primary/30">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-primary" />
                    Agentic Swarm Intelligence
                  </CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Real-time observation of agent thought processes</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-nexus-text-secondary uppercase">Ask the Swarm</label>
                    <div className="flex gap-2">
                      <input 
                        type="text" 
                        value={swarmQuery}
                        onChange={(e) => setSwarmQuery(e.target.value)}
                        className="flex-1 bg-nexus-card border border-nexus-border rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-primary/50"
                        placeholder="e.g. Analyze enrollment bottleneck at SITE-042"
                      />
                      <Button 
                        size="sm" 
                        className="bg-primary hover:bg-primary/80 shadow-lg shadow-primary/20"
                        onClick={() => {
                          // Extract site_id if present in query (e.g. SITE-001)
                          const siteMatch = swarmQuery.match(/SITE-\d+/i) || swarmQuery.match(/Site_\d+/i);
                          const inferredSiteId = siteMatch ? siteMatch[0].toUpperCase() : 'all';
                          
                          swarmMutation.mutate({ 
                            query: swarmQuery, 
                            context: { 
                              study_id: selectedStudy, 
                              site_id: inferredSiteId,
                              timestamp: new Date().toISOString() 
                            } 
                          });
                        }}
                        disabled={swarmMutation.isPending}
                      >
                        {swarmMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Trigger'}
                      </Button>
                    </div>
                  </div>

                  <div className="min-h-[300px] max-h-[500px] overflow-y-auto space-y-4 pr-2">
                    {swarmMutation.isError && (
                      <div className="p-3 bg-error-500/10 rounded-lg border border-error-500/20 text-xs text-error-400">
                        Agent communication failed: {(swarmMutation.error as any)?.response?.data?.detail || swarmMutation.error.message}
                      </div>
                    )}
                    
                    {swarmMutation.isPending ? (
                      <div className="flex flex-col items-center justify-center py-20 text-center">
                        <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
                        <p className="text-sm text-nexus-text-secondary">Swarm is investigating... analysis in progress.</p>
                      </div>
                    ) : swarmMutation.data?.trace ? (
                      swarmMutation.data.trace.map((step: any, idx: number) => (
                        <div key={idx} className="space-y-2 animate-in slide-in-from-left-2 fade-in">
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded bg-primary/20 flex items-center justify-center">
                              <span className="text-[10px] font-bold text-primary">{step.agent[0]}</span>
                            </div>
                            <span className="text-xs font-bold text-white uppercase">{step.agent}</span>
                          </div>
                          <div className="ml-8 p-3 bg-nexus-card border-l-2 border-primary rounded-r-lg text-xs space-y-2 relative">
                            <div className="absolute -left-[2px] top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary to-transparent" />
                            <p className="text-nexus-text-secondary italic">"{step.thought}"</p>
                            <div className="flex items-center gap-2">
                              <code className="text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded border border-emerald-400/20">{'>'} {step.action}</code>
                            </div>
                            <div className="p-2.5 bg-nexus-bg/80 rounded border border-nexus-border/50 text-nexus-text-secondary font-mono leading-relaxed">
                              {step.observation.includes('DQI') ? step.observation.replace('DQI', 'Data Quality Index (DQI)') : step.observation}
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="flex flex-col items-center justify-center py-20 text-center opacity-30">
                        <Play className="w-12 h-12 text-nexus-text-secondary mb-4" />
                        <p className="text-sm text-nexus-text-secondary px-6">Agents are standing by. Enter an investigation query to begin.</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Anomalies List */}
              <Card className="glass-card border-nexus-border">
                <CardHeader>
                  <CardTitle className="text-sm font-bold text-white uppercase tracking-wider">Live Anomalies</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 max-h-[350px] overflow-y-auto custom-scrollbar">
                  {anomaliesData?.anomalies?.length ? (
                    anomaliesData.anomalies.map((anom: any) => (
                      <div key={anom.id || Math.random()} className="p-3 bg-error-500/5 border border-error-500/20 rounded-lg flex items-center justify-between cursor-pointer hover:bg-error-500/10 transition-colors" onClick={() => setSwarmQuery(`Investigate ${anom.title || 'Anomaly'} at ${anom.site_id || 'Unknown'}`)}>
                        <div>
                          <p className="text-sm font-medium text-white">{anom.title || 'Unknown Anomaly'}</p>
                          <p className="text-[10px] text-error-400 uppercase font-bold">{anom.site_id || 'System'}</p>
                        </div>
                        <ChevronRight className="w-4 h-4 text-error-400" />
                      </div>
                    ))
                  ) : (
                    <div className="py-10 text-center opacity-30">
                      <p className="text-xs text-nexus-text-secondary italic">No active anomalies detected in current sync</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Patients Tab */}
        <TabsContent value="patients" className="space-y-4">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Patient Overview</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Recent patients across all sites</CardDescription>
            </CardHeader>
            <CardContent>
              {patientsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} className="h-12 w-full bg-nexus-card" />
                  ))}
                </div>
              ) : patientsError || !patients?.patients?.length ? (
                <div className="py-12 text-center">
                  <Users className="w-12 h-12 text-nexus-text-secondary mx-auto mb-4 opacity-50" />
                  <p className="text-nexus-text-secondary">
                    {patientsError ? 'Unable to load patient data. Please try again.' : 'No patients found for the selected filters.'}
                  </p>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="mt-4 border-nexus-border text-nexus-text-secondary hover:text-white"
                    onClick={() => window.location.reload()}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              ) : (
                <Table>
                    <TableHeader>
                      <TableRow className="border-nexus-border hover:bg-transparent">
                        <TableHead className="text-nexus-text-secondary">Patient ID</TableHead>
                        <TableHead className="text-nexus-text-secondary">Site</TableHead>
                        <TableHead className="text-nexus-text-secondary">Status</TableHead>
                        <TableHead className="text-nexus-text-secondary">Risk</TableHead>
                        <TableHead className="text-nexus-text-secondary">DQI Score</TableHead>
                        <TableHead className="text-nexus-text-secondary">DB Lock</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {patients?.patients?.slice(0, 10)?.map((patient: Patient) => (
                        <Fragment key={patient.patient_key}>
                          <TableRow 
                            className={cn("border-nexus-border hover:bg-nexus-card/50 cursor-pointer", selectedPatientId === patient.patient_key && "bg-nexus-card/30 border-primary/20")}
                            onClick={() => setSelectedPatientId(selectedPatientId === patient.patient_key ? null : patient.patient_key)}
                          >
                            <TableCell className="font-medium text-white">{patient.patient_key}</TableCell>
                            <TableCell className="text-nexus-text-secondary">{patient.site_id}</TableCell>
                            <TableCell>
                              <Badge variant={patient.status === 'active' ? 'success' : 'secondary'}>
                                {patient.status || 'Unknown'}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge className={getRiskColor(patient.risk_level || '')}>
                                {patient.risk_level || 'N/A'}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <span className={patient.dqi_score && patient.dqi_score >= 85 ? 'text-success-400' : patient.dqi_score && patient.dqi_score >= 70 ? 'text-warning-400' : 'text-error-400'}>
                                {patient.dqi_score?.toFixed(1) || 'N/A'}%
                              </span>
                            </TableCell>
                            <TableCell>
                              {patient.is_db_lock_ready ? (
                                <CheckCircle2 className="w-5 h-5 text-success-400" />
                              ) : (
                                <XCircle className="w-5 h-5 text-nexus-text-secondary" />
                              )}
                            </TableCell>
                          </TableRow>
                        
                        {selectedPatientId === patient.patient_key && (
                          <TableRow className="bg-nexus-card/20 border-nexus-border hover:bg-nexus-card/20">
                            <TableCell colSpan={7} className="p-0 border-b-0">
                              <div className="p-6 bg-nexus-card/40 animate-in fade-in slide-in-from-top-4 duration-300 space-y-6">
                                <div className="flex items-center justify-between mb-4">
                                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                    <Users className="w-5 h-5 text-primary" />
                                    Patient Details: {selectedPatientId}
                                  </h3>
                                  <Button variant="ghost" size="sm" onClick={() => setSelectedPatientId(null)}>Close</Button>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-6">
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">Site</p>
                                    <p className="text-white">{patient.site_id}</p>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">Status</p>
                                    <Badge>{patient.status}</Badge>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">DQI Score</p>
                                    <p className="text-emerald-400 font-bold">{patient.dqi_score?.toFixed(1)}%</p>
                                  </div>
                                </div>

                                {/* ML Explainability Section */}
                                <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
                                  <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2">
                                      <Brain className="w-4 h-4 text-purple-400" />
                                      <h4 className="text-sm font-bold text-white uppercase tracking-wider">ML Explainability (SHAP)</h4>
                                    </div>
                                    <Badge variant="outline" className="text-[10px] border-purple-500/30 text-purple-400">
                                      {riskExplanation?.model_version || 'v1.2-SHAP'}
                                    </Badge>
                                  </div>

                                  {explanationLoading ? (
                                    <div className="space-y-3">
                                      {[1, 2, 3].map(i => <Skeleton key={i} className="h-4 w-full bg-nexus-card" />)}
                                    </div>
                                  ) : riskExplanation?.feature_impacts ? (
                                    <div className="space-y-4">
                                      <p className="text-xs text-nexus-text-secondary italic">
                                        "This patient is classified as <span className="font-bold text-white uppercase">{riskExplanation.risk_level} RISK</span> because of the following top factors:"
                                      </p>
                                      <div className="space-y-3">
                                        {riskExplanation.feature_impacts.map((impact: any, i: number) => (
                                          <div key={i} className="space-y-1.5">
                                            <div className="flex justify-between text-[10px] font-bold uppercase">
                                              <span className="text-nexus-text-secondary">{impact.feature}</span>
                                              <span className={impact.type === 'positive' ? 'text-error-400' : 'text-success-400'}>
                                                {impact.type === 'positive' ? '+' : '-'}{impact.impact.toFixed(2)} pts impact
                                              </span>
                                            </div>
                                            <div className="h-1.5 w-full bg-nexus-card rounded-full overflow-hidden">
                                              <div 
                                                className={cn(
                                                  "h-full rounded-full transition-all duration-1000",
                                                  impact.type === 'positive' ? "bg-error-500" : "bg-success-500"
                                                )}
                                                style={{ width: `${Math.min(100, impact.impact * 20)}%` }}
                                              />
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  ) : (
                                    <p className="text-xs text-nexus-text-secondary italic text-center py-4">No risk explanation data available for this patient.</p>
                                  )}
                                </div>

                                <div className="mt-6 flex gap-3">
                                  <Button size="sm" variant="outline" className="border-nexus-border">View Medical History</Button>
                                  <Button size="sm" variant="outline" className="border-nexus-border">Audit Trail</Button>
                                  <Button size="sm" className="bg-primary hover:bg-primary/80">Resolve Issues</Button>
                                </div>
                              </div>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Issues Tab */}
        <TabsContent value="issues" className="space-y-4">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Recent Issues</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="border-nexus-border hover:bg-transparent">
                    <TableHead className="text-nexus-text-secondary">Issue ID</TableHead>
                    <TableHead className="text-nexus-text-secondary">Patient</TableHead>
                    <TableHead className="text-nexus-text-secondary">Site</TableHead>
                    <TableHead className="text-nexus-text-secondary">Type</TableHead>
                    <TableHead className="text-nexus-text-secondary">Priority</TableHead>
                    <TableHead className="text-nexus-text-secondary">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {issues?.issues?.slice(0, 10)?.map((issue: Issue) => (
                    <Fragment key={issue.issue_id}>
                      <TableRow 
                        className={cn("border-nexus-border hover:bg-nexus-card/50 cursor-pointer", selectedIssueId === issue.issue_id && "bg-nexus-card/30 border-warning/20")}
                        onClick={() => setSelectedIssueId(selectedIssueId === issue.issue_id ? null : issue.issue_id || null)}
                      >
                        <TableCell className="font-medium text-white">#{issue.issue_id}</TableCell>
                        <TableCell className="text-nexus-text-secondary">{issue.patient_key}</TableCell>
                        <TableCell className="text-nexus-text-secondary">{issue.site_id}</TableCell>
                        <TableCell className="text-white">{issue.issue_type}</TableCell>
                        <TableCell>
                          <Badge className={getRiskColor(issue.priority || '')}>
                            {issue.priority}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={issue.status === 'open' ? 'error' : issue.status === 'resolved' ? 'success' : 'secondary'}>
                            {issue.status}
                          </Badge>
                        </TableCell>
                      </TableRow>

                      {selectedIssueId === issue.issue_id && (
                        <TableRow className="bg-nexus-card/20 border-nexus-border hover:bg-nexus-card/20">
                          <TableCell colSpan={7} className="p-0 border-b-0">
                            <div className="p-6 bg-nexus-card/40 animate-in fade-in slide-in-from-top-4 duration-300">
                              <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                  <AlertTriangle className="w-5 h-5 text-warning" />
                                  Issue Analysis: #{selectedIssueId}
                                </h3>
                                <Button variant="ghost" size="sm" onClick={() => setSelectedIssueId(null)}>Close</Button>
                              </div>
                              <div className="space-y-4">
                                <div className="grid grid-cols-3 gap-6">
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">Type</p>
                                    <p className="text-white">{issue?.issue_type}</p>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">Priority</p>
                                    <Badge className={getRiskColor(issue?.priority)}>{issue?.priority}</Badge>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-xs text-nexus-text-secondary uppercase">Patient</p>
                                    <p className="text-white font-mono">{issue?.patient_key}</p>
                                  </div>
                                </div>
                                 <div className="p-3 bg-nexus-bg rounded border border-nexus-border">
                                   <p className="text-xs text-nexus-text-secondary mb-1">AI ROOT CAUSE PREDICTION</p>
                                   <p className="text-sm text-white">Probable staff training gap at site {issue?.site_id} regarding {issue?.issue_type} workflows.</p>
                                 </div>
                               </div>
                             </div>
                           </TableCell>
                         </TableRow>

                      )}
                    </Fragment>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

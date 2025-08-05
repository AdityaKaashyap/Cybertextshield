import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  StatusBar,
  ActivityIndicator,
  SafeAreaView,
  Platform,
  Linking,
  Share,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import * as Animatable from 'react-native-animatable';
import Toast from 'react-native-toast-message';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/MaterialIcons';

const API_BASE_URL = 'http://10.0.2.2:8000'; // For Android emulator
// const API_BASE_URL = 'http://localhost:8000'; // For iOS simulator
// const API_BASE_URL = 'http://YOUR_IP:8000'; // For physical device

const App = () => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isOnline, setIsOnline] = useState(true);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    loadHistory();
    checkApiHealth();
  }, []);

  const loadHistory = async () => {
    try {
      const savedHistory = await AsyncStorage.getItem('scanHistory');
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const saveToHistory = async (newResult) => {
    try {
      const newHistory = [newResult, ...history.slice(0, 19)]; // Keep last 20 results
      setHistory(newHistory);
      await AsyncStorage.setItem('scanHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  };

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setIsOnline(data.status === 'healthy');
    } catch (error) {
      setIsOnline(false);
      console.error('API health check failed:', error);
    }
  };

  const analyzeSMS = async () => {
    if (!message.trim()) {
      Toast.show({
        type: 'error',
        text1: 'Error',
        text2: 'Please enter a message to analyze',
      });
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const resultData = {
        ...data,
        timestamp: new Date().toISOString(),
        id: Date.now(),
      };

      setResult(resultData);
      saveToHistory(resultData);

      Toast.show({
        type: data.is_spam ? 'error' : 'success',
        text1: data.is_spam ? '🚨 Spam Detected!' : '✅ Safe Message',
        text2: `Confidence: ${(data.confidence * 100).toFixed(1)}%`,
      });

    } catch (error) {
      console.error('Error analyzing message:', error);
      Toast.show({
        type: 'error',
        text1: 'Analysis Failed',
        text2: 'Unable to connect to the server. Please check your connection.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearMessage = () => {
    setMessage('');
    setResult(null);
  };

  const clearHistory = async () => {
    Alert.alert(
      'Clear History',
      'Are you sure you want to clear all scan history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            setHistory([]);
            await AsyncStorage.removeItem('scanHistory');
            Toast.show({
              type: 'success',
              text1: 'History Cleared',
              text2: 'All scan history has been removed',
            });
          },
        },
      ]
    );
  };

  const shareResult = async () => {
    if (!result) return;

    const shareText = `SMS Spam Detection Result:\n\nMessage: "${result.message}"\nResult: ${result.prediction.toUpperCase()}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\n\nScanned with SMS Spam Detector`;

    try {
      await Share.share({
        message: shareText,
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const openSettings = () => {
    Alert.alert(
      'Settings',
      'Configure API endpoint and other settings',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'API Settings',
          onPress: () => {
            // You can implement API endpoint configuration here
            Toast.show({
              type: 'info',
              text1: 'Feature Coming Soon',
              text2: 'API configuration will be available in future updates',
            });
          },
        },
      ]
    );
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getResultIcon = (isSpam) => {
    return isSpam ? '🚨' : '✅';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#e74c3c';
    if (confidence > 0.6) return '#f39c12';
    return '#27ae60';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#667eea" />
      
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>SMS Spam Detector</Text>
          <Text style={styles.headerSubtitle}>
            {isOnline ? '🟢 Online' : '🔴 Offline'}
          </Text>
        </View>
        
        <View style={styles.headerButtons}>
          <TouchableOpacity
            style={styles.headerButton}
            onPress={() => setShowHistory(!showHistory)}
          >
            <Icon name="history" size={24} color="#fff" />
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.headerButton}
            onPress={openSettings}
          >
            <Icon name="settings" size={24} color="#fff" />
          </TouchableOpacity>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content}>
        {!showHistory ? (
          <Animatable.View animation="fadeInUp" duration={800}>
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Enter SMS Message</Text>
              <TextInput
                style={styles.textInput}
                value={message}
                onChangeText={setMessage}
                placeholder="Type or paste the SMS message here..."
                multiline
                numberOfLines={4}
                placeholderTextColor="#999"
              />
              
              <View style={styles.buttonContainer}>
                <TouchableOpacity
                  style={[styles.button, styles.clearButton]}
                  onPress={clearMessage}
                  disabled={!message.trim()}
                >
                  <Icon name="clear" size={20} color="#666" />
                  <Text style={styles.clearButtonText}>Clear</Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={[styles.button, styles.analyzeButton]}
                  onPress={analyzeSMS}
                  disabled={isLoading || !message.trim()}
                >
                  {isLoading ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Icon name="security" size={20} color="#fff" />
                  )}
                  <Text style={styles.analyzeButtonText}>
                    {isLoading ? 'Analyzing...' : 'Analyze'}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>

            {result && (
              <Animatable.View animation="fadeInUp" duration={600} style={styles.resultContainer}>
                <View style={styles.resultHeader}>
                  <Text style={styles.resultTitle}>Analysis Result</Text>
                  <TouchableOpacity onPress={shareResult}>
                    <Icon name="share" size={24} color="#667eea" />
                  </TouchableOpacity>
                </View>
                
                <View style={[styles.resultCard, result.is_spam ? styles.spamCard : styles.hamCard]}>
                  <View style={styles.resultIcon}>
                    <Text style={styles.resultEmoji}>{getResultIcon(result.is_spam)}</Text>
                  </View>
                  
                  <View style={styles.resultContent}>
                    <Text style={styles.resultLabel}>
                      {result.is_spam ? 'SPAM DETECTED' : 'SAFE MESSAGE'}
                    </Text>
                    
                    <View style={styles.confidenceContainer}>
                      <Text style={styles.confidenceLabel}>Confidence:</Text>
                      <Text style={[styles.confidenceValue, { color: getConfidenceColor(result.confidence) }]}>
                        {(result.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                    
                    <Text style={styles.timestamp}>
                      Scanned: {formatTimestamp(result.timestamp)}
                    </Text>
                  </View>
                </View>
                
                {result.is_spam && (
                  <View style={styles.warningContainer}>
                    <Icon name="warning" size={20} color="#e74c3c" />
                    <Text style={styles.warningText}>
                      This message appears to be spam. Be cautious of links, requests for personal information, or urgent actions.
                    </Text>
                  </View>
                )}
              </Animatable.View>
            )}
          </Animatable.View>
        ) : (
          <Animatable.View animation="fadeInUp" duration={600}>
            <View style={styles.historyContainer}>
              <View style={styles.historyHeader}>
                <Text style={styles.historyTitle}>Scan History</Text>
                <TouchableOpacity onPress={clearHistory}>
                  <Icon name="delete" size={24} color="#e74c3c" />
                </TouchableOpacity>
              </View>
              
              {history.length === 0 ? (
                <View style={styles.emptyHistory}>
                  <Icon name="history" size={48} color="#ccc" />
                  <Text style={styles.emptyHistoryText}>No scan history yet</Text>
                </View>
              ) : (
                history.map((item) => (
                  <View key={item.id} style={styles.historyItem}>
                    <View style={styles.historyItemHeader}>
                      <Text style={styles.historyItemEmoji}>{getResultIcon(item.is_spam)}</Text>
                      <Text style={[styles.historyItemLabel, item.is_spam ? styles.spamLabel : styles.hamLabel]}>
                        {item.is_spam ? 'SPAM' : 'SAFE'}
                      </Text>
                      <Text style={styles.historyItemConfidence}>
                        {(item.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                    
                    <Text style={styles.historyItemMessage} numberOfLines={2}>
                      {item.message}
                    </Text>
                    
                    <Text style={styles.historyItemTimestamp}>
                      {formatTimestamp(item.timestamp)}
                    </Text>
                  </View>
                ))
              )}
            </View>
          </Animatable.View>
        )}
      </ScrollView>
      
      <Toast />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.8,
    marginTop: 4,
  },
  headerButtons: {
    flexDirection: 'row',
  },
  headerButton: {
    marginLeft: 16,
    padding: 8,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  inputContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 16,
    fontSize: 16,
    color: '#333',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    flex: 0.48,
    justifyContent: 'center',
  },
  clearButton: {
    backgroundColor: '#f8f9fa',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  clearButtonText: {
    color: '#666',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
  analyzeButton: {
    backgroundColor: '#667eea',
  },
  analyzeButtonText: {
    color: '#fff',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
  resultContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
  },
  resultCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
  },
  spamCard: {
    backgroundColor: '#fee',
    borderColor: '#e74c3c',
    borderWidth: 1,
  },
  hamCard: {
    backgroundColor: '#efe',
    borderColor: '#27ae60',
    borderWidth: 1,
  },
  resultIcon: {
    marginRight: 16,
  },
  resultEmoji: {
    fontSize: 32,
  },
  resultContent: {
    flex: 1,
  },
  resultLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: '#333',
    marginBottom: 8,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
    marginRight: 8,
  },
  confidenceValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  timestamp: {
    fontSize: 12,
    color: '#999',
  },
  warningContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 12,
    backgroundColor: '#ffeaa7',
    borderRadius: 8,
    borderColor: '#e17055',
    borderWidth: 1,
  },
  warningText: {
    flex: 1,
    marginLeft: 8,
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  historyContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
  },
  emptyHistory: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyHistoryText: {
    fontSize: 16,
    color: '#999',
    marginTop: 12,
  },
  historyItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    paddingVertical: 12,
  },
  historyItemHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  historyItemEmoji: {
    fontSize: 20,
    marginRight: 8,
  },
  historyItemLabel: {
    fontSize: 14,
    fontWeight: '700',
    flex: 1,
  },
  spamLabel: {
    color: '#e74c3c',
  },
  hamLabel: {
    color: '#27ae60',
  },
  historyItemConfidence: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  historyItemMessage: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
    lineHeight: 20,
  },
  historyItemTimestamp: {
    fontSize: 12,
    color: '#999',
  },
});

export default App;